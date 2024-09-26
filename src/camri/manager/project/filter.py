from dataclasses import dataclass
from typing import Union, List, Optional


@dataclass
class Replace:
    """
    Contains information related to a replacement operation.

    Attributes:
        regex (str): The regular expression to be matched.
        replacement (str): The string that will replace the matched expression.
    """
    regex: str
    replacement: str


@dataclass
class Filter:
    subject: Union[List[str], str, None] = None
    session: Union[List[str], str, None] = None
    modal: Union[List[str], str, None] = None
    annotation: Union[List[str], str, None] = None
    regex: Optional[str] = None
    regex_ignore: Optional[str] = None
    ext:  Optional[str] = None


@dataclass
class Modifier:
    replace: Optional[Replace] = None,
    prefix: Optional[str] = None, 
    suffix: Optional[str] = None,
    ext: Optional[str] = None,
    absdir: Union[bool, str] = False