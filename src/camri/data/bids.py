"""A dependency-light BIDS/derivatives file index."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

_ENTITY_RE = re.compile(r"^(?P<key>[A-Za-z0-9]+)-(?P<value>[^_]+)$")
_KNOWN_KEYS = {
    "sub": "subject",
    "ses": "session",
    "task": "task",
    "acq": "acquisition",
    "ce": "ce",
    "rec": "reconstruction",
    "dir": "direction",
    "run": "run",
    "echo": "echo",
    "part": "part",
    "space": "space",
    "res": "resolution",
    "den": "density",
    "desc": "description",
    "label": "label",
}


@dataclass(frozen=True)
class BidsFile:
    path: Path
    entities: Mapping[str, str] = field(default_factory=dict)
    suffix: str = ""
    extension: str = ""
    datatype: str | None = None

    def __getattr__(self, name: str):
        if name in self.entities:
            return self.entities[name]
        raise AttributeError(name)


class BidsDataset:
    def __init__(self, root: str | Path, *, include_derivatives: bool = False) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self.include_derivatives = include_derivatives
        self._index: tuple[BidsFile, ...] | None = None

    def _iter_paths(self) -> Iterator[Path]:
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            relative_parts = path.relative_to(self.root).parts
            if not self.include_derivatives and relative_parts and relative_parts[0] == "derivatives":
                continue
            yield path

    @property
    def files(self) -> tuple[BidsFile, ...]:
        if self._index is None:
            self._index = tuple(self._parse(path) for path in self._iter_paths())
        return self._index

    def _parse(self, path: Path) -> BidsFile:
        name = path.name
        if name.endswith(".nii.gz"):
            stem, extension = name[:-7], ".nii.gz"
        else:
            suffix_path = Path(name)
            extension = suffix_path.suffix
            stem = name[: -len(extension)] if extension else name
        tokens = stem.split("_")
        suffix = tokens.pop() if tokens else ""
        entities: dict[str, str] = {}
        for token in tokens:
            match = _ENTITY_RE.match(token)
            if not match:
                continue
            key, value = match.group("key"), match.group("value")
            entities[_KNOWN_KEYS.get(key, key)] = value
        datatype = None
        relative = path.relative_to(self.root)
        for parent in reversed(relative.parts[:-1]):
            if parent not in {"anat", "func", "dwi", "fmap", "perf", "pet", "meg", "eeg", "ieeg", "beh", "derivatives"}:
                datatype = parent
                break
            if parent in {"anat", "func", "dwi", "fmap", "perf", "pet", "meg", "eeg", "ieeg", "beh"}:
                datatype = parent
                break
        return BidsFile(path, entities, suffix, extension, datatype)

    def query(
        self,
        *,
        subject: str | Iterable[str] | None = None,
        session: str | Iterable[str] | None = None,
        datatype: str | None = None,
        suffix: str | None = None,
        extension: str | Iterable[str] | None = None,
        **entities: str | Iterable[str] | None,
    ) -> list[BidsFile]:
        filters = {"subject": subject, "session": session, "datatype": datatype, "suffix": suffix, "extension": extension, **entities}

        def matches(actual, expected, key: str) -> bool:
            if expected is None:
                return True
            expected_values = (expected,) if isinstance(expected, str) else tuple(expected)
            if key in {"subject", "session"}:
                actual = actual.removeprefix(f"{key[:3]}-") if isinstance(actual, str) else actual
                expected_values = tuple(str(v).removeprefix(f"{key[:3]}-") for v in expected_values)
            return str(actual) in {str(v) for v in expected_values}

        return [
            item
            for item in self.files
            if all(matches(item.entities.get(key) if key not in {"datatype", "suffix", "extension"} else getattr(item, key), expected, key) for key, expected in filters.items())
        ]

    def validate_basic(self) -> list[str]:
        errors: list[str] = []
        subjects = [p for p in self.root.iterdir() if p.is_dir() and p.name.startswith("sub-")]
        if not subjects:
            errors.append("dataset contains no sub-* directories")
        for item in self.files:
            if item.path.name.endswith(".nii.gz") and not item.suffix:
                errors.append(f"missing BIDS suffix: {item.path}")
        return errors


class Derivatives:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)

    @property
    def pipelines(self) -> list[str]:
        return sorted(path.name for path in self.root.iterdir() if path.is_dir())

    def dataset(self, pipeline: str) -> BidsDataset:
        return BidsDataset(self.root / pipeline, include_derivatives=True)


class Project:
    def __init__(self, root: str | Path, *, raw: BidsDataset | None = None, derivatives: Derivatives | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self.raw = raw or BidsDataset(self.root)
        derivatives_path = self.root / "derivatives"
        self.derivatives = derivatives or (Derivatives(derivatives_path) if derivatives_path.is_dir() else None)

    @classmethod
    def open(cls, root: str | Path) -> Project:
        return cls(root)

    def create_derivative(self, name: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError("invalid derivative name")
        path = self.root / "derivatives" / name
        path.mkdir(parents=True, exist_ok=True)
        self.derivatives = Derivatives(self.root / "derivatives")
        return path


class PyBidsAdapter:
    """Optional adapter around pybids without importing it in core CAMRI."""

    def __init__(self, root: str | Path, **kwargs) -> None:
        try:
            from bids import BIDSLayout
        except ImportError as exc:
            raise ImportError("install camri[bids] to use PyBidsAdapter") from exc
        self.layout = BIDSLayout(str(root), **kwargs)

    def query(self, **kwargs):
        return self.layout.get(return_type="filename", **kwargs)


class LegacyCamriLayout:
    """Read the former data/proc/mask layout through the new query interface."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)

    def query(self, *, area: str | None = None, extension: str | None = None) -> list[Path]:
        base = self.root / area if area else self.root
        if not base.is_dir():
            raise FileNotFoundError(base)
        paths = sorted(path for path in base.rglob("*") if path.is_file())
        if extension:
            paths = [path for path in paths if path.name.endswith(extension)]
        return paths


__all__ = ["BidsDataset", "BidsFile", "Derivatives", "LegacyCamriLayout", "Project", "PyBidsAdapter"]
