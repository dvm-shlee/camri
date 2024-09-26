from __future__ import annotations
import os, re
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
from .filter import Replace
from .helper import *


@dataclass
class DataItem:
    """
    Contains structured data information.
    
    Attributes:
        by_depth (Dict[int, Dict[str, List[str]]]): A dictionary with depth as the key and a dictionary of session path and file names as the value.
    """
    by_depth: Dict[int, Dict[str, List[str]]]


@dataclass
class FileItem:
    """
    Contains information related to a file.

    Attributes:
        subject (str): The subject associated with the file.
        session (Optional[str]): The session associated with the file. Can be None.
        modal (Optional[str]): The modal associated with the file. Can be None.
        annotation (Optional[str]): The annotation associated with the file. Can be None.
        absdir (str): The absolute directory where the file is located.
        filename (str): The name of the file.
        fileext (str): The extension of the file.

    Methods:
        abspath: Return the absolute path of the file.
        basename: Return the base name of the file.
        modify: Modify the name of the file.
        match: Check if the filename matches a regular expression.
        is_match: Check if the filename matches a regular expression.
        has_ext: Check if the file has a specific extension.
    """
    subject: str
    session: Optional[str]
    modal: Optional[str]
    annotation: Optional[str]
    absdir: str
    filename: str
    fileext: str

    @property
    def abspath(self) -> str:
        return os.path.join(self.absdir, self.basename)

    @property
    def basename(self) -> str:
        return f"{self.filename}.{self.fileext}"

    def modify(
        self, 
        replace: Optional[Replace] = None,
        prefix: Optional[str] = None, 
        suffix: Optional[str] = None,
        ext: Optional[str] = None,
        absdir: Union[bool, str] = False,
        noext: bool = False
        ) -> str:
        
        if all([e == None for e in [prefix, suffix, replace, ext, noext]]):
            if absdir:
                if isinstance(absdir, bool):
                    return self.abspath
                else:
                    return os.path.join(absdir, self.basename)
            else:
                return self.basename
        else:
            modified = str(self.filename)
            if replace:
                compiled = re.compile(replace.regex)
                modified = compiled.sub(replace.replacement, modified)
            if prefix:
                modified = f"{prefix}{modified}"
            if suffix:
                modified = f"{modified}{suffix}"
            if ext:
                ext = ".".join(strip_empty_str_in_list(ext.split(".")))
            else:
                ext = self.fileext
            
            if not noext:
                modified = f"{modified}.{ext}"
            if absdir:
                if isinstance(absdir, bool):
                    return os.path.join(self.absdir, modified)
                else:
                    os.makedirs(absdir, exist_ok=True)
                    return os.path.join(absdir, modified)
            else:
                return modified

    def match(self, regex: str) -> Union[dict, List[str], None]:
        groups = re.compile(regex)
        matched = groups.match(self.filename)
        if matched:
            if groups.groupindex:
                return matched.groupdict()
            else:
                return matched.groups()
        else:
            return None
        
    def is_match(self, regex: str) -> Union[dict, List[str], None]:
        groups = re.compile(regex)
        matched = groups.match(self.filename)
        if matched:
            return True
        else:
            return False

    def has_ext(self, ext: str) -> bool:
        this_ext = strip_empty_str_in_list(self.fileext.split("."))
        quer_ext = strip_empty_str_in_list(ext.split('.'))
        return this_ext == quer_ext

    def __repr__(self):
        path_list = self.absdir.split(os.sep)
        if self.session == None:
            depth = 3
        else:
            depth = 4
        path = os.sep.join(path_list[-1*depth:])
        
        repr = f"FileItem('{path}', '{self.basename}')"
        return repr
        
    def __str__(self):
        return self.abspath
        

@dataclass
class SessionItem:
    """
    Contains information related to a session.

    Attributes:
        subject (str): The subject associated with the session.
        session (Optional[str]): The specific session. Can be None.
        files (Union[List[FileInfo], Dict[str, List[FileInfo]]]): The files associated with the session. Can be a list or a dictionary.

    Methods:
        length: Return the number of files in the session.
    """
    subject: str
    session: Optional[str]
    files: Union[List[FileItem], Dict[str, List[FileItem]]]

    @property
    def length(self) -> Union[int, Dict[str, int]]:
        if isinstance(self.files, dict):
            return {modal:len(finfos) for modal, finfos in self.files.items()}
        else:
            return len(self.files)
        
    def __repr__(self):
        if isinstance(self.length, dict):
            files = ", ".join([f"{modal}:[n={length}]" for modal, length in self.length.items()])
            files = f'{{{files}}}'
        else:
            files = f"n={self.length}"
        contents = [self.subject, files]
        if self.session != None:
            contents.insert(1, self.session)
        contents = ", ".join(contents)
        return f"SessionItem({contents})"
    