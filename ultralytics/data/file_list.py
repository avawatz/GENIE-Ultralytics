from collections.abc import Sequence
import os

class FilesList(Sequence):
    def __init__(self, files: dict, project_dir: str):
        assert os.path.isdir(project_dir)
        self.files, self.aug_idx = self._merge_dict(files)
        self.project_dir = os.path.abspath(project_dir)

    def _merge_dict(self, files: dict):
        out = list()
        for name in list(files.keys()):
            if name == "augmented_set":
                continue
            else:
                out.extend(files.pop(name))
        if files.get("augmented_set") is None:
            aug_idx = -1
        else:
            aug_idx = len(out)
            out.extend(files.pop('augmented_set'))
        return out, aug_idx

    def __getitem__(self, idx, ann=False):
        if idx < 0:
            raise IndexError("Negative indexing is not allowed")
        name, _ = os.path.splitext(self.files[idx])
        if self.aug_idx != -1 and idx >= self.aug_idx:
            return (os.path.join(self.project_dir, "augmented_set", self.files[idx]),
                    os.path.join(self.project_dir, "annotations", f"{name}.json"))
        else:
            return (os.path.join(self.project_dir, "images", self.files[idx]),
                    os.path.join(self.project_dir, "annotations", f"{name}.json"))


    def __len__(self):
        return len(self.files)
