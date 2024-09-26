import os
from typing import Any, Optional
from .filter import Replace
from .dataset import RawDataset, ProcDataset, StepItem, MaskItem

ignores = ['.ipynb_checkpoints']

class Project:
    """TODO: config file"""
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self._scan()

    def _scan(self):
        self.dataclass = {}
        for d in os.listdir(self.path):
            """hard coded"""
            if d == "data":
                try:
                    self.dataclass[d] = RawDataset(os.path.join(self.path, d), ignores=ignores)
                except:
                    # no rawdata
                    self.dataclass[d] = None
            elif d == "proc":
                self.dataclass[d] = ProcDataset(os.path.join(self.path, d), ignores=ignores)
            elif d == "mask":
                self.dataclass[d] = ProcDataset(os.path.join(self.path, d), mask=True, ignores=ignores)
            elif d == "done":
                pass

    def get_path(self, dataclass: str) -> str:
        return os.path.join(self.path, dataclass)

    def _init_step(self, id: str, name: str, annotation: str) -> str:
        """
        id: code
        annotation: name
        """
        info = StepItem(id, name, annotation, None)
        step_path = os.path.join(self.get_path('proc'), info.fullname)
        os.makedirs(step_path, exist_ok=True)
        return step_path
    
    def _init_mask(self, modal: str, acq: Optional[str] = None) -> str:
        info = MaskItem(modal=modal, acq=acq)
        mask_path = os.path.join(self.get_path('mask'), info.fullname)
        os.makedirs(mask_path, exist_ok=True)
        return mask_path

    def __getattr__(self, __name: str) -> Any:
        return self.dataclass[__name]

    def reload(self):
        self._scan()
    
    def __repr__(self):
        self.reload()
        repr = []
        repr.append(f"* path: {self.path}\n")
        for d, ds in sorted(self.dataclass.items()):
            repr.append(f"{d}: {str(ds)}")
        return "\n".join(repr)