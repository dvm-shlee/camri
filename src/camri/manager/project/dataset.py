from __future__ import annotations
import os
import re
from dataclasses import dataclass
from .parser import BaseParser
from .reference import Inherits, MaxDepthRef
from .items import DataItem
from typing import Optional, Union, List, Any



@dataclass
class StepItem:
    """
    Contains information related to a step in a pipeline.

    Attributes:
        id (str): The id of the step.
        name (str): The name of the step.
        annotation (str): The annotation of the step.
        dataset (Optional[StepDataset]): The dataset associated with the step. Can be None.

    Methods:
        path: Return the path of the step.
    """
    id: str
    name: str
    annotation: str
    dataset: Optional[StepDataset] = None

    @property
    def fullname(self) -> str:
        return f"{self.id}_{self.title}"
    
    @property
    def title(self) -> str:
        return f"{self.name}_{self.annotation}"

    @property
    def path(self) -> str:
        return self.dataset.path

    def filter(self, *args: Any, **kwargs: Any) -> StepDataset:
        return self.dataset.filter(*args, **kwargs)
    
    def __call__(self, *args: Any, **kwargs: Any) -> StepDataset:
        return self.filter(*args, **kwargs)
    
    def __repr__(self):
        return f"StepItem(id='{self.id}', title='{f'{self.title}'}')"
    

@dataclass
class MaskItem:
    """
    Contains information related to a mask.

    Attributes:
        modal (str): The modality of the mask.
        acq (str): Acquisition method
        dataset (Optional[StepDataset]): The dataset associated with the mask. Can be None.
    """
    modal: str
    acq: Optional[str] = None
    dataset: Optional[StepDataset] = None

    @property
    def fullname(self) -> str:
        if self.acq:
            path = f'{self.modal}-{self.acq}'
        else:
            path = self.modal
        return path
    
    def filter(self, *args, **kwargs):
        return self.dataset.filter(*args, **kwargs)


class RawDataset(BaseParser):
    """
    Class for handling and parsing NIP-RawDataset specifications which correspond to BIDS standards.

    Attributes:
        path (str): The root path of the dataset to parse.
        inherits (Optional[Inherits]): Optional Inherits object with information to be inherited by a new instance of a class.

    Methods:
        __init__: Initialize the RawDataset instance.
        _validate: Validate the dataset structure according to BIDS standards.
        _construct: Construct session_list and file_list.
        filter: Filter data based on several criteria.
    """
    def __init__(self, 
                 path: str, 
                 validate: bool = True,
                 inherits: Optional[Inherits] = None,
                 *args, **kwargs):
        """
        Initialize the StepDataset instance.

        Parameters:
        path (str): The root path of the dataset to parse.
        validate (bool, optional): If True, validation of the dataset structure is performed on initialization. Default is True.
        inherits (Inherits, optional): Optional Inherits object with information to be inherited by a new instance of a class.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(path, *args, **kwargs)
        if validate:
            self._validate()
        if inherits:
            self._inherits(inherits)
        else:
            self._construct()

    def _validate(self):
        """
        Validate the dataset structure according to the provided reference and modality.
        """
        dirs = DataItem(self._dirs_by_depth)
        files = DataItem(self._files_by_depth)
        ref = MaxDepthRef(2, 3)
        self.sessions, self.subjects, self.modals = \
            self._validator(dirs, files, self.max_depth, ref, modal=True)

    def _construct(self):
        """
        Construct session_list and file_list from the dataset.
        """
        files = DataItem(self._files_by_depth)
        ref = MaxDepthRef(2, 3)
        modal = True
        
        self.session_list, self.file_list = self._constructor(files, 
                                                              ref, 
                                                              modal)
        
    def filter(self, 
               subject: Union[List[str], str, None] = None, 
               session: Union[List[str], str, None] = None,
               modal: Union[List[str], str, None] = None,  # dataclass?
               annotation: Union[List[str], str, None] = None,
               regex: Optional[str] = None,
               regex_ignore: Optional[str] = None,
               ext: Optional[str] = None) -> RawDataset:
        """
        Filter the dataset based on various parameters such as subject, session, modality, annotation, regex, regex_ignore, and extension.

        Parameters:
        subject (Union[List[str], str, None], optional): The subject(s) to filter by.
        session (Union[List[str], str, None], optional): The session(s) to filter by.
        modal (Union[List[str], str, None], optional): The modality(ies) to filter by.
        annotation (Union[List[str], str, None], optional): The annotation(s) to filter by.
        regex (Optional[str], optional): A regex pattern to filter by.
        regex_ignore (Optional[str], optional): A regex pattern to ignore while filtering.
        ext (Optional[str], optional): The file extension to filter by.

        Returns:
        RawDataset: A new instance of StepDataset with filtered data.
        """
        file_list, sess_list = self._filter(self.file_list, 
                                            self.session_list,
                                            subject,
                                            session,
                                            modal,
                                            annotation,
                                            regex,
                                            regex_ignore,
                                            ext)
        
        inherits = Inherits(self.subjects, self.sessions, self.modals,
                            file_list, sess_list)
        return RawDataset(self.path, False, inherits)
            

class StepDataset(BaseParser):
    """
    Class for handling and parsing NIP-ProcDataset specifications. These specifications consist of process 
    step folders containing BIDS-like datasets which do not contain modality subdirectories (only subjects or 
    subjects-sessions).

    Attributes:
        path (str): The root path of the dataset to parse.
        inherits (Optional[Inherits]): Optional Inherits object with information to be inherited by a new instance of a class.

    Methods:
        __init__: Initialize the StepDataset instance.
        _validate: Validate the dataset structure.
        _construct: Construct session_list and file_list.
        filter: Filter data based on several criteria.
    """
    def __init__(self, 
                 path: str, 
                 validate: bool = True, 
                 inherits: Optional[Inherits] = None,
                 *args, **kwargs):
        """
        Initialize the StepDataset instance.

        Parameters:
        path (str): The root path of the dataset to parse.
        validate (bool, optional): If True, validation of the dataset structure is performed on initialization. Default is True.
        inherits (Inherits, optional): Optional Inherits object with information to be inherited by a new instance of a class.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(path, *args, **kwargs)
        if validate:
            self._validate()
        if inherits:
            self._inherits(inherits)
        else:
            self._construct()

    def _validate(self):
        """
        Validate the dataset structure according to the provided reference and modality.
        """
        dirs = DataItem(self._dirs_by_depth)
        files = DataItem(self._files_by_depth)
        ref = MaxDepthRef(1, 2)
        self.sessions, self.subjects, self.modals = \
            self._validator(dirs, files, self.max_depth, ref, modal=False)

    def _construct(self):
        """
        Construct session_list and file_list from the dataset.
        """
        files = DataItem(self._files_by_depth)
        ref = MaxDepthRef(1, 2)
        modal = False
        
        self.session_list, self.file_list = self._constructor(files, 
                                                              ref, 
                                                              modal)

    def filter(self, 
               subject: Union[List[str], str, None] = None, 
               session: Union[List[str], str, None] = None,
               modal: Union[List[str], str, None] = None,  # dataclass?
               annotation: Union[List[str], str, None] = None,
               regex: Optional[str] = None,
               regex_ignore: Optional[str] = None,
               ext: Optional[str] = None) -> StepDataset:
        """
        Filter the dataset based on various parameters such as subject, session, modality, annotation, regex, regex_ignore, and extension.

        Parameters:
        subject (Union[List[str], str, None], optional): The subject(s) to filter by.
        session (Union[List[str], str, None], optional): The session(s) to filter by.
        modal (Union[List[str], str, None], optional): The modality(ies) to filter by.
        annotation (Union[List[str], str, None], optional): The annotation(s) to filter by.
        regex (Optional[str], optional): A regex pattern to filter by.
        regex_ignore (Optional[str], optional): A regex pattern to ignore while filtering.
        ext (Optional[str], optional): The file extension to filter by.

        Returns:
        StepDataset: A new instance of StepDataset with filtered data.
        """
       
        file_list, sess_list = self._filter(self.file_list, 
                                            self.session_list,
                                            subject,
                                            session,
                                            modal,
                                            annotation,
                                            regex,
                                            regex_ignore,
                                            ext)
        
        inherits = Inherits(self.subjects, self.sessions, self.modals,
                            file_list, sess_list)
        return StepDataset(self.path, False, inherits)


class ProcDataset:
    """
    Class for processing a dataset.

    Attributes:
        path (str): The root path of the dataset to process.
        is_mask (bool): Flag indicating if the dataset is a mask.

    Methods:
        __init__: Initialize the ProcDataset instance.
        _parse: Parse the dataset.
        _subpath: Return a subpath within the dataset.
        _scan_subpath: Scan a subpath and return a StepDataset.
        get_dataset: Return a dataset based on a provided target.
        avail: Return a list of available datasets.
    """
    def __init__(self, path: str, mask: bool = False, ignores: Optional[List[str]] = None):
        self.path = path
        self.ignores = ignores
        self.is_mask = mask
        self._parse()

    def _parse(self):
        if self.is_mask:
            self.mask_list = []
            mask_pattern = re.compile(r"(?P<modal>[a-zA-Z0-9]+)(-(?P<acq>[a-zA-Z0-9]+))?")
            for modal_str in os.listdir(self.path):
                if os.path.isdir(self._subpath(modal_str)):
                    if matched := mask_pattern.match(modal_str):
                        si = matched.groupdict()
                        try:
                            self.mask_list.append(MaskItem(modal=si['modal'], acq=si['acq'], dataset=self._scan_subpath(modal_str)))
                        except:
                            # empty
                            pass
        else:
            self.step_list = []
            step_pattern = re.compile(r"(?P<id>[a-zA-Z0-9]{4})_(?P<name>[a-zA-Z0-9\-]+)_(?P<annotation>[a-zA-Z0-9\-]+)")
            for step in os.listdir(self.path):
                if os.path.isdir(self._subpath(step)):
                    if matched := step_pattern.match(step):
                        si = matched.groupdict()
                        step_path = os.path.join(self.path, step)
                        try:
                            self.step_list.append(StepItem(si["id"], si["name"], si["annotation"], dataset=self._scan_subpath(step)))
                        except:
                            subpath_files = [files for _, _, files in os.walk(step_path) if len(files)]
                            if not len(subpath_files) == 0:
                                from warnings import warn
                                warn(f"Folder '{step}' does not comply with the BIDS standard format.")


    def __call__(self, id: str, **kwargs) -> Union[StepItem, 
                                                   MaskItem, 
                                                   List[StepItem], 
                                                   List[MaskItem], 
                                                   None]:
        if self.is_mask:
            list_obj = self.mask_list
            dsets = [dset for dset in list_obj if dset.modal == id]
        else:
            list_obj = self.step_list
            dsets = [dset for dset in list_obj if dset.id == id]
        if kwargs:
            for k, v in kwargs.items():
                dsets = [dset for dset in dsets if getattr(dset, k) == v]
        if len(dsets) == 0:
            return None
        elif len(dsets) == 1:
            return dsets.pop()
        else:
            return dsets

    def _subpath(self, path):
        return os.path.join(self.path, path)

    def _scan_subpath(self, path) -> StepDataset:
        return StepDataset(path=os.path.join(self.path, path), ignores=self.ignores)

    @property
    def avail(self):
        if self.is_mask:
            return self.mask_list
        else:
            return self.step_list
        
    def __repr__(self):
        if self.is_mask:
            title = "Available mask dataset:\n"
        else:
            title = "Available processing step dataset:\n"
        return title + '\n'.join(sorted([' + '+str(e) for e in self.avail]))