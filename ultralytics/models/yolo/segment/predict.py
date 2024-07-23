# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.results import Results, Masks
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops
import torch
from ultralytics.cfg import get_cfg


class SegmentationPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a segmentation model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.segment import SegmentationPredictor

        args = dict(model='yolov8n-seg.pt', source=ASSETS)
        predictor = SegmentationPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes the SegmentationPredictor with the provided configuration, overrides, and callbacks."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"


    def _calculate_grad_emb(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        nms_preds = ops.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        pred_dict = {"batch_idx": [],
                     "masks": [],
                     "bboxes": [],
                     "cls": []}

        proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]  # tuple if PyTorch model or array if exported
        for batch_idx, pred in enumerate(nms_preds):
            orig_img = orig_imgs[batch_idx]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask(proto[batch_idx], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)
            img_path = self.batch[0][batch_idx]
            for _ in range(pred.shape[0]):
                pred_dict['batch_idx'].append(batch_idx)
            if pred.shape[0] >= 1:
                pred_dict['masks'].extend(masks)
                pred_dict['cls'].append(pred[:, -1])
                pred_dict["bboxes"].append(pred[:, :4])
            # else:
            #     pred_dict['segments'].append([])
            #     pred_dict['cls'].append([])

            # print(masks.xyn)
            # pred_dict['segments'].append([torch.tensor(maskmasks.xyn)
        del nms_preds, pred
        torch.cuda.empty_cache()
        
        pred_dict['batch_idx'] = torch.tensor(pred_dict['batch_idx'],dtype=torch.long).to(self.device)
        pred_dict['cls'] = torch.cat(pred_dict['cls'], dim=0).to(self.device)
        pred_dict['masks'] = torch.stack(pred_dict['masks'], dim=0).to(self.device)
        pred_dict['bboxes'] = torch.cat(pred_dict['bboxes'], dim=0).to(self.device)

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
        nc = self.model.model.model[-1].nc
        reg_max = self.model.model.model[-1].reg_max
        no = nc + reg_max * 4
        # print(nc, reg_max, no)

        # print(preds)
        # print(preds[0].grad_fn)
        # print(dir(self.model))
        # print(self.model.model.model[-1].reg_max)
        # print(preds[0].shape)
        # print([i.shape for i in preds[-1]])
        # print(self.device)
        # print(self.model.model.nc)
        # feats = preds[-1]
        # print([xi.view(feats[0].shape[0], no, -1).shape for xi in feats])

        # print(pred_distri.shape, temp.shape, pred_distri.grad_fn)
        loss, _ = self.model.model.loss(preds=tuple(preds), batch=pred_dict)

        grad = torch.autograd.grad(loss, preds[-1][0])
        _, grad = torch.cat([xi.view(grad[0].shape[0], no, -1) for xi in grad], 2).split(
            (reg_max * 4, nc), 1
        )
        # print(pred_conf.shape)
        del preds
        torch.cuda.empty_cache()
        # grad = torch.cat(grad, dim=0).detach()
        assert grad.shape[-2] == self.model.model.nc
        # print([grad].shape)
        # grad = torch.mean(grad, axis=-1)
        # print("PIT ", grad.shape)
        # print("OUT: ", torch.mean(grad, dim=-1))
        print(grad.reshape(grad.shape[0], -1).shape)

        return grad.detach()


    def switch_model_gradient(self, enable: bool):
        for p in self.model.parameters():
            if p is enable:
                break
            p.requires_grad = enable   


    def postprocess(self, preds, img, orig_imgs):
        """Applies non-max suppression and processes detections for each image in an input batch."""
        
        if self.grad_emb:
            return self._calculate_grad_emb(preds, img, orig_imgs)

        p, probs = ops.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            classes=self.args.classes,
            return_probs=True
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]  # tuple if PyTorch model or array if exported
        assert len(proto) == len(probs)
        for i, (pred, prob) in enumerate(zip(p, probs)):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            print(pred.shape)
            if not len(pred):  # save empty boxes
                masks = None
            elif self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks, all_probs=prob))

        if not len(results):
            orig_img = orig_imgs[0]
            img_path = self.batch[0][0]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=None,
            all_probs=torch.tensor([])))

        return results
