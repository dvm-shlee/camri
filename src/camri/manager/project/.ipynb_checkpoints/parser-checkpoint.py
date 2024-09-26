import os
import re
from copy import copy
from typing import Optional, List, Tuple, Union
from .items import DataItem, FileItem, SessionItem
from .reference import MaxDepthRef, Inherits
from .error import *
from .helper import *


class BaseParser:
    """
    This class is designed to parse files and directories at a given path. It enables the processing of neuroimaging datasets in
    a structured and flexible manner.

    It can be used as a standalone parser or as a base class for more specific parsers. It provides support for parsing both files
    and directories, depth tracking, flexible filtering, and BIDS validation. The parser also implements .nipignore for better
    control over the parsing process.
    
    Public Attributes:
    - dir_only (bool): If True, only directories will be parsed.
    - max_depth (int): The maximum depth of the directory tree.
    - subjects (Optional[List[str]]): List of subjects found in the dataset.
    - sessions (Optional[List[str]]): List of sessions found in the dataset.
    - modals (Optional[List[str]]): List of modalities found in the dataset.
    - file_list (Optional[List[FileInfo]]): List of files parsed in the dataset.
    - session_list (Optional[List[SessionInfo]]): List of sessions found in the dataset.
    
    Private Attributes:
    - _abs_path (str): The absolute path of the root.
    - _abs_path_list (List[str]): The list representation of the absolute path.
    - _abs_path_depth (int): The total depth of the root path.
    - _files_by_depth (dict): A dictionary mapping from depth to another dictionary, which maps from relative 
                                  paths to filenames.
    - _dirs_by_depth (dict): A dictionary mapping from depth to another dictionary, which maps from relative 
                                 paths to directory names.
    """
    
    # public properties
    dir_only: bool
    max_depth: int
    subjects: Optional[List[str]]
    sessions: Optional[List[str]]
    modals: Optional[List[str]]
    file_list = Optional[List[FileItem]]
    session_list = Optional[List[SessionItem]]
    
    # private properties
    _abs_path: str
    _abs_path_list: List[str]
    _abs_path_depth: int
    _files_by_depth: dict
    _dirs_by_depth: dict

    """
    This class is used to parse files and dirs at a specific path.

    TODO: implement .nipignore for the project
    """
    def __init__(self, path: str = None, dir_only: bool = False, ignores: Optional[List[str]] = None):
        """
        Initialize the BaseParser instance.
        
        Parameters:
        - path (str): The root path of the dataset to parse. Default is None.
        - dir_only (bool): If True, only directories will be parsed. Default is False.
        """
        self.path = path
        self.dir_only = dir_only
        self.ignores = ignores
        self.parse()
    
    def parse(self):
        self._init_process()
        self._parse_all()

    def _init_process(self):
        """
        Initialize the process by setting up private-level attributes for the root path.

        Attributes:
        - _abs_path (str): The absolute path of the root.
        - _abs_path_list (list): The list representation of the absolute path.
        - _abs_path_depth (int): The total depth of the root path.
        """
        abs_path = os.path.abspath(self.path)
        abs_path_list = strip_empty_str_in_list(abs_path.split(os.path.sep))
        abs_path_depth = len(abs_path_list)
        
        #store to class's private attributes
        self._abs_path = abs_path
        self._abs_path_list = abs_path_list
        self._abs_path_depth = abs_path_depth
        
    def _parse_all(self):
        """
        Parse all subdirectories and files, store them in private attributes, and measure the maximum depth.

        Private Attributes:
        - _files_by_depth (dict): A dictionary mapping from depth to another dictionary, which maps from relative 
                                  paths to filenames.
        - _dirs_by_depth (dict): A dictionary mapping from depth to another dictionary, which maps from relative 
                                 paths to directory names.

        Public Attributes:
        - max_depth (int): The maximum depth of the directory tree.
        """
        max_depth = 0
        files_by_depth = dict()
        dirs_by_depth = dict()
        self._ignored = dict(dirs=[], files=[])
        for (dirpath, dirnames, filenames) in os.walk(self._abs_path):
            dirpath = str_to_list(dirpath)
            relpath_list = dirpath[self._abs_path_depth:]
            relpath = os.path.sep.join(relpath_list)
            depth_step = len(relpath_list)
            filtered = self._filter_dataset(files=filenames, dirs=dirnames)

            dirnames = filtered['filtered']['dirs']
            filenames = filtered['filtered']['files']
            self._ignored['dirs'].extend([os.path.join(self.path, relpath, d) for d in filtered['ignored']['dirs']])
            self._ignored['files'].extend([os.path.join(self.path, relpath, f) for f in filtered['ignored']['files']])

            if len(dirnames) == 0:
                # update max depth
                if depth_step > max_depth:
                    max_depth = depth_step
            else:
                if depth_step not in dirs_by_depth.keys():
                    dirs_by_depth[depth_step] = dict()
                dirs_by_depth[depth_step][relpath] = sorted(dirnames)

            if not self.dir_only:
                if len(filenames):
                    if depth_step not in files_by_depth.keys():
                        files_by_depth[depth_step] = dict()    
                    files_by_depth[depth_step][relpath] = sorted(filenames)
        self._files_by_depth = files_by_depth
        self._dirs_by_depth = dirs_by_depth
        self.max_depth = max_depth
    
    def _filter_dataset(self, files: Optional[List[str]] = None, dirs: Optional[List[str]] = None) -> dict:
        ignored_files = []
        ignored_dirs = []
        if self.ignores:
            filtered_files = []
            filtered_dirs = []   
            if files:
                for f in files:
                    if f in self.ignores:
                        ignored_files.append(f)
                    else:
                        filtered_files.append(f)
            if dirs:
                for d in dirs:
                    if d in self.ignores:
                        ignored_dirs.append(d)
                    else:
                        filtered_dirs.append(d)
        else:
            filtered_files = files
            filtered_dirs = dirs
        return dict(filtered={'files': filtered_files,
                              'dirs': filtered_dirs},
                    ignored={'files': ignored_files,
                             'dirs': ignored_dirs})



    @classmethod
    def _validator(cls, 
                   dirs: DataItem, 
                   files: DataItem, 
                   max_depth: int, 
                   ref: MaxDepthRef,
                   modal: bool = False,
                   dir_only: bool = False) -> Tuple[List[str], List[str], List[str]]:
        """
        Validate the dataset structure according to BIDS standards.

        The method checks the depth of the dataset (based on the reference given for single and multi session dataset) and the compliance 
        of subject, session(optional), modal, and file names to BIDS naming conventions. In case of non-compliance, an InvalidFormatError is raised.

        The validated subjects and sessions are stored in the `subjects` and `sessions` attributes of the instance.

        Raises:
        - ValueError: If the dataset depth is not compliant with BIDS standards.
        - InvalidFormatError: If subject, session, or file names are not compliant with BIDS naming conventions.
        """
        # check dataset depth, 2 for single session and 3 for multi session dataset
        if max_depth not in ref.list:
            raise ValueError(f"Invalid dataset depth: {max_depth}. Expected depth is {ref.single_session} "
                             f"for single session or {ref.multi_session} for multi session dataset.")
        
        # define regular expressions for BIDS naming conventions
        subj_pattern = re.compile(r'sub-[a-zA-Z0-9]+')
        sess_pattern = re.compile(r'ses-[a-zA-Z0-9]+')
        file_pattern = re.compile(r'sub-[a-zA-Z0-9]+(_ses-[a-zA-Z0-9]+)?.*')
        
        # validate subject names
        subjects = sorted([s for s in dirs.by_depth[0]['']])
        is_subjects = [subj_pattern.match(s) != None for s in subjects]
        if not all(is_subjects):
            raise InvalidFormatError("Not all subjects match the expected 'sub-*' format.")
        
        # if multi session dataset, validate session names
        if max_depth == ref.multi_session:
            sessions = sorted(list(set([sess for sesses in dirs.by_depth[1].values() for sess in sesses])))
            is_sessions = [sess_pattern.match(s) != None for s in sessions]
            if not all(is_sessions):
                raise InvalidFormatError("Not all sessions match the expected 'ses-*' format.")
        else:
            sessions = None

        # if files are included in parsing, validate file names
        if not dir_only:
            filenames = sorted([filename for (_, sess) in files.by_depth[max_depth].items() for filename in sess])
            is_not_bidsfiles = [filename for filename in filenames if file_pattern.match(filename) == None]
            if len(is_not_bidsfiles):
                files_need_to_be_reviewed = "\n".join(is_not_bidsfiles)
                from warnings import warn
                warn(f"Not all file match the expected file format.\n{files_need_to_be_reviewed}")
        
        # returns validated subjects and sessions
        if modal:
            modals = sorted(list(set([modal for modals in dirs.by_depth[max_depth-1].values() for modal in modals])))
        else:
            modals = None
        return subjects, sessions, modals
        
    def _inherits(self, inherits: Inherits):
        """
        A dataclass to carry over attributes when generating a new instance of a class.

        This class is designed to store the required information when generating a new instance
        with filtered file_list and session_list.

        Attributes:
        - subjects (List[str]): A list of subjects in the dataset.
        - sessions (List[str]): A list of sessions in the dataset.
        - modal (Optional[List[str]]): An optional list of modalities in the dataset. This could be None.
        - file_list (List[FileInfo]): A list of FileInfo objects representing the files in the dataset.
        - session_list (List[SessionInfo]): A list of SessionInfo objects representing the sessions in the dataset.
        """
        self.sessions = inherits.sessions
        self.subjects = inherits.subjects
        self.modals = inherits.modal
        self.file_list = inherits.file_list
        self.session_list = inherits.session_list
    
    def _constructor(self,
                     files: DataItem,
                     ref: MaxDepthRef,
                     modal: bool = False) -> Tuple[List[SessionItem], List[FileItem]]:
        """
        Process the data and construct session and file lists.

        Parameters:
        - files (DataInfo): Data containing the file paths.
        - ref (MaxDepthRef): Reference to the maximum depth.
        - modal (bool): If True, the modal attribute will be considered. Default is False.

        Returns:
        - Tuple: A tuple containing a list of SessionInfo objects and a list of FileInfo objects.
        """
    
        # prep spaceholders
        processed = []
        session_list = []
        file_list = []
        
        # loop over the paths at max depth
        for sess_path, filenames in files.by_depth[self.max_depth].items():
            
            # parse meta data
            if self.max_depth == ref.single_session:
                if modal:
                    subj, modal = str_to_list(sess_path)
                else:
                    subj = str_to_list(sess_path)[0]
                    modal = None
                sess = None
            elif self.max_depth == ref.multi_session:
                if modal:
                    subj, sess, modal = str_to_list(sess_path)
                else:
                    subj, sess = str_to_list(sess_path)
                    modal = None
            else:
                raise ValueError(f"Invalid dataset depth: {self.max_depth}. Expected depth is {ref.single_session} "
                                 f"for single session or {ref.multi_session} for multi session dataset.")

            # construct session and file lists
            task_id = f"{subj}-{sess}"
            if task_id in processed:
                sinfo = [si for si in session_list if si.subject == subj and si.session == sess][0]
            else:
                if modal:
                    sinfo = SessionItem(subj, sess, {})
                else:
                    sinfo = SessionItem(subj, sess, [])
                session_list.append(sinfo)
                processed.append(task_id)
            for f in filenames:
                filename, fileext = f.split('.', 1)
                finfo = FileItem(subj, sess, modal, self._abs_path_list[-1], 
                                 os.path.join(self._abs_path, sess_path), filename, fileext)
                if modal:
                    if modal not in sinfo.files.keys():
                        sinfo.files[modal] = []
                    sinfo.files[modal].append(finfo)
                    file_list.append(finfo)
                else:
                    sinfo.files.append(finfo)
                    file_list.append(finfo)
            
            if modal:
                sinfo.files[modal] = sorted(sinfo.files[modal], key=lambda x: (x.filename, x.fileext))
            else:        
                sinfo.files = sorted(sinfo.files, key=lambda x: (x.modal, x.filename, x.fileext))
        
        # sort constructed lists and returns
        session_list = sorted(session_list, key=lambda x: (x.subject, x.session))
        file_list = sorted(file_list, key=lambda x: (x.subject, x.session, x.modal, x.filename, x.fileext))
        return session_list, file_list

    @classmethod
    def _filter(cls,
                file_list: List[FileItem],
                session_list: List[SessionItem],
                subject: Union[List[str], str, None] = None, 
                session: Union[List[str], str, None] = None,
                modal: Union[List[str], str, None] = None,
                annotation: Union[List[str], str, None] = None,
                regex: Optional[str] = None,
                regex_ignore: Optional[str] = None,
                ext: Optional[str] = None):
        """
        Filter the file list and session list based on the specified criteria.

        Parameters:
        - file_list (List[FileInfo]): List of FileInfo objects to filter.
        - session_list (List[SessionInfo]): List of SessionInfo objects to filter.
        - subject (Union[List[str], str, None]): Subject(s) to filter by. Default is None.
        - session (Union[List[str], str, None]): Session(s) to filter by. Default is None.
        - modal (Union[List[str], str, None]): Modal(s) to filter by. Default is None.
        - annotation (Union[List[str], str, None]): Annotation(s) to filter by. Default is None.
        - regex (Optional[str]): Regex pattern to filter by. Default is None.
        - regex_ignore (Optional[str]): Regex pattern to ignore. Default is None.
        - ext (Optional[str]): File extension to filter by. Default is None.

        Returns:
        - filtered_file_list, filtered_sess_list: Filtered lists of FileInfo and SessionInfo objects.
        """
        filtered_file_list = copy(file_list)
        filtered_sess_list = copy(session_list)
        if subject:
            if isinstance(subject, str):
                subject = [subject]
            filtered_file_list = [finfo for finfo in filtered_file_list if finfo.subject in subject]
            filtered_sess_list = [sinfo for sinfo in filtered_sess_list if sinfo.subject in subject]
        if session:
            if isinstance(session, str):
                session = [session]
            filtered_file_list = [finfo for finfo in filtered_file_list if finfo.session in session]
            filtered_sess_list = [sinfo for sinfo in filtered_sess_list if sinfo.session in session]
        if modal:
            if isinstance(modal, str):
                modal = [modal]
            filtered_file_list = [finfo for finfo in filtered_file_list if finfo.modal in modal]
        if annotation:
            if isinstance(annotation, str):
                annotation = [annotation]
            filtered_file_list = [finfo for finfo in filtered_file_list if finfo.annotation in annotation]
        if ext:
            filtered_file_list = [finfo for finfo in filtered_file_list if finfo.has_ext(ext)]
        if regex:
            filtered_file_list = [finfo for finfo in filtered_file_list if finfo.is_match(regex)]
        if regex_ignore:
            filtered_file_list = [finfo for finfo in filtered_file_list if not finfo.is_match(regex_ignore)]
        
        filtered_sess_list = cls._session_filter(filtered_sess_list,
                                                 modal,
                                                 annotation,
                                                 regex,
                                                 regex_ignore,
                                                 ext)
        
        return filtered_file_list, filtered_sess_list
    
    @classmethod
    def _session_filter(cls, 
                        session_list: List[SessionItem],
                        modal: Union[List[str], str, None] = None,
                        annotation: Union[List[str], str, None] = None,
                        regex: Optional[str] = None, 
                        regex_ignore: Optional[str] = None, 
                        ext: Optional[str] = None):
        """
        Filter the session list based on the specified criteria.

        Parameters:
        - session_list (List[SessionInfo]): List of SessionInfo objects to filter.
        - modal (Union[List[str], str, None]): Modal(s) to filter by. Default is None.
        - annotation (Union[List[str], str, None]): Annotation(s) to filter by. Default is None.
        - regex (Optional[str]): Regex pattern to filter by. Default is None.
        - regex_ignore (Optional[str]): Regex pattern to ignore. Default is None.

        Returns:
        - filtered_sess_list: Filtered list of SessionInfo objects.
        """
        session_list = [copy(s) for s in session_list]
        for sess in session_list:   
            if modal:
                if isinstance(modal, str):
                    modal = [modal]
                if isinstance(sess.files, dict):
                    sess.files = {m:[f for f in fs if f.modal in modal] for m, fs in sess.files.items()}
                else:
                    sess.files = [finfo for finfo in sess.files if finfo.modal in modal]
            if annotation:
                if isinstance(annotation, str):
                    annotation = [annotation]
                if isinstance(sess.files, dict):
                    sess.files = {m:[f for f in fs if f.annotation in annotation] for m, fs in sess.files.items()}
                else:
                    sess.files = [finfo for finfo in sess.files if finfo.annotation in annotation]
            if regex:
                if isinstance(sess.files, dict):
                    sess.files = {m:[f for f in fs if f.is_match(regex)] for m, fs in sess.files.items()}
                else:
                    sess.files = [finfo for finfo in sess.files if finfo.is_match(regex)]
            if regex_ignore:
                if isinstance(sess.files, dict):
                    sess.files = {m:[f for f in fs if not f.is_match(regex_ignore)] for m, fs in sess.files.items()}
                else:
                    sess.files = [finfo for finfo in sess.files if not finfo.is_match(regex_ignore)]
            if ext:
                if isinstance(sess.files, dict):
                    sess.files = {m:[f for f in fs if f.has_ext(ext)] for m, fs in sess.files.items()}
                else:
                    sess.files = [finfo for finfo in sess.files if finfo.has_ext(ext)]
            if isinstance(sess.files, dict):
                sess.files = {k:v for k, v in sess.files.items() if len(v)}
            
        return session_list