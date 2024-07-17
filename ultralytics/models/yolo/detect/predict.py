# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results, Probs
from ultralytics.utils import ops
from ultralytics.cfg import get_cfg
from ultralytics.utils import LOGGER
import torch
import numpy as np

class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model='yolov8n.pt', source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """
    

    def _calculate_grad_emb(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        nms_preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        pred_dict = {"batch_idx": [],
                     "bboxes": [],
                     "cls": []}
        for batch_idx, pred in enumerate(nms_preds):
            orig_img = orig_imgs[batch_idx]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][batch_idx]
            for _ in range(pred.shape[0]):
                pred_dict['batch_idx'].append(batch_idx)
            pred_dict['bboxes'].append(pred[:, :4])
            pred_dict['cls'].append(pred[:, -1])
        del nms_preds, pred
        torch.cuda.empty_cache()
        
        pred_dict['batch_idx'] = torch.tensor(pred_dict['batch_idx'],dtype=torch.long).to(self.device)
        if len('bboxes') >= 2:
            pred_dict['bboxes'] = torch.cat(pred_dict['bboxes'], dim=0).to(self.device)
            pred_dict['cls'] = torch.cat(pred_dict['cls'], dim=0).to(self.device)
        elif len('bboxes') == 1:
            pred_dict['bboxes'] = pred_dict['bboxes'][0].to(self.device)
            pred_dict['cls'] = pred_dict['cls'][0].to(self.device)
        else:
            pred_dict['bboxes'] = torch.tensor([]).to(self.device)
            pred_dict['cls'] = torch.tensor([]).to(self.device)
        
        if isinstance(self.model.model.args, dict):
            if not self.model.model.args.get('box'):
                self.model.model.args['box'] = 7.5
            if not self.model.model.args.get('cls'):
                self.model.model.args['cls'] = 0.5
            if not self.model.model.args.get('dfl'):
                self.model.model.args['dfl'] = 1.5
            self.model.model.args = get_cfg(self.model.model.args)
        else:
            if not hasattr(self.model.model.args, 'box'):
                setattr(self.model.model.args, 'box', 7.5)
            if not hasattr(self.model.model.args, 'cls'):
                setattr(self.model.model.args, 'cls', 0.5)
            if not hasattr(self.model.model.args, 'dfl'):
                setattr(self.model.model.args, 'dfl', 1.5)

        self.model.model.end2end=False
        # print(self.model.model.model[-1].reg_max)
        print(preds[0].shape)
        print([i.shape for i in preds[-1]])
        # print(self.device)
        
        
        
        loss, _ = self.model.model.loss(preds=tuple(preds), batch=pred_dict)
        grad = torch.autograd.grad(loss, preds[-1][0])
        del preds
        torch.cuda.empty_cache()
        grad = torch.cat(grad, dim=0).detach()
        assert grad.shape[-2] == self.model.model.nc
        grad = torch.mean(grad, axis=-1)
        # print("PIT ", grad.shape)
        # print("OUT: ", torch.mean(grad, dim=-1))

        return grad


    def switch_model_gradient(self, enable: bool):
        for p in self.model.parameters():
            if p is enable:
                break
            p.requires_grad = enable    


    @torch.enable_grad
    def stream_grad_emb(self, source=None, model=None, *args, **kwargs):
        """Streams real-time inference on camera feed and saves results to file."""
        if self.args.verbose:
            LOGGER.info("")

        # Setup model
        if not self.model:
            self.setup_model(model)
        self.switch_model_gradient(enable=True)

        with self._lock:  # for thread-safe inference
            # Setup source every time predict is called
            self.setup_source(source if source is not None else self.args.source)

            # Check if save_dir/ label file exists
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # Warmup model
            if not self.done_warmup:
                self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                self.done_warmup = True

            self.seen, self.windows, self.batch = 0, [], None
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )
            self.run_callbacks("on_predict_start")
            for self.batch in self.dataset:
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch
                print(len(paths))

                # Preprocess
                with profilers[0]:
                    im = self.preprocess(im0s)

                # Inference
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)
                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                        continue

                # Postprocess
                with profilers[2]:
                    self.results = self.postprocess(preds, im, im0s)
                # self.run_callbacks("on_predict_postprocess_end")

                # # Visualize, save, write results
                # n = len(im0s)
                # for i in range(n):
                #     self.seen += 1
                #     self.results[i].speed = {
                #         "preprocess": profilers[0].dt * 1e3 / n,
                #         "inference": profilers[1].dt * 1e3 / n,
                #         "postprocess": profilers[2].dt * 1e3 / n,
                #     }
                #     if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                #         s[i] += self.write_results(i, Path(paths[i]), im, s)

                # Print batch results
                # if self.args.verbose:
                #     LOGGER.info("\n".join(s))

                self.run_callbacks("on_predict_batch_end")
                yield from self.results
            else:
                self.switch_model_gradient(enable=False)
                print("CLEARED")
                self.results = None
                torch.cuda.empty_cache()


    def __call__(self, source=None, model=None, stream=False, grad_emb=False, *args, **kwargs):
        """Performs inference on an image or stream."""
        self.stream = stream
        self.grad_emb = grad_emb
        print("GRAD_EMB: ", grad_emb)
        if stream:
            if grad_emb:
                return self.stream_grad_emb(source, model, *args, **kwargs)
            else:
                return self.stream_inference(source, model, *args, **kwargs)
        else:
            if grad_emb:
                grad_out = list(self.stream_grad_emb(source, model, *args, **kwargs))
                # print("OUT", grad_out[0].shape)
                # if len(grad_out) > 1:
                return torch.stack(grad_out, dim=0)
                # else:
                #     for i in grad_out:
                #         print(i.shape)
                #     return grad_out
            else:
                return list(self.stream_inference(source, model, *args, **kwargs))  # merge list of Result into one


    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        
        if self.grad_emb:
            return self._calculate_grad_emb(preds, img, orig_imgs)
        # for tensor in preds:
        #     if isinstance(tensor, torch.Tensor):
        #         print(tensor.shape)
        #     else:
        #         for i in tensor:
        #             print(i.shape)
        preds, probs = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
            return_probs=True
        )
        #for i in probs:
        #    print(i.shape, i.max(1, keepdim=True), torch.sum(i, dim=1))
        # for tensor in preds:
        #     if isinstance(tensor, torch.Tensor):
        #         print(tensor.shape)
        #     else:
        #         for i in tensor:
        #             print(i.shape)
        # for i in range(5):
        #     print(preds[0][:, 5:, i])

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        assert len(preds) == len(probs)

        for i, (pred, prob) in enumerate(zip(preds, probs)):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred,
            all_probs=prob))

        if not len(results):
            orig_img = orig_imgs[0]
            img_path = self.batch[0][0]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=None,
            all_probs=torch.tensor([])))

        print("RESULTS: ", len(results))

        return results