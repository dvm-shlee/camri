from dataclasses import dataclass
from typing import Optional, List
from .items import FileItem, SessionItem

@dataclass
class MaxDepthRef:
    """
    Contains reference values for maximum depth.

    Attributes:
        single_session (int): The reference value for a single session.
        multi_session (int): The reference value for a multi-session.

    Methods:
        list: Returns a list containing the single_session and multi_session values.
    """
    single_session: int
    multi_session: int
    
    @property
    def list(self):
        return [self.single_session, self.multi_session]
    

@dataclass
class Inherits:
    """
    Contains information to be inherited by a new instance of a class.

    Attributes:
        subjects (List[str]): A list of subjects.
        sessions (List[str]): A list of sessions.
        modal (Optional[List[str]]): A list of modalities. Can be None.
        file_list (List[FileInfo]): A list of FileInfo objects.
        session_list (List[SessionInfo]): A list of SessionInfo objects.
    """
    subjects: List[str]
    sessions: List[str]
    modal: Optional[List[str]]
    file_list: List[FileItem]
    session_list: List[SessionItem]