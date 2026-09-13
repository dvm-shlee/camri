"""Portable job specifications shared by all CAMRI execution backends."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import IO


class JobState(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Resources:
    """Resource requests understood by local containers and schedulers."""

    cpus: int = 1
    memory: str | None = None
    gpus: int | str | None = None
    time_limit: str | None = None

    def __post_init__(self) -> None:
        if self.cpus < 1:
            raise ValueError("cpus must be at least 1")
        if isinstance(self.gpus, int) and self.gpus < 0:
            raise ValueError("gpus must be non-negative")


@dataclass(frozen=True)
class Mount:
    """A host path mounted into a container."""

    source: Path | str
    target: Path | str
    read_only: bool = False

    def __post_init__(self) -> None:
        if not str(self.source):
            raise ValueError("mount source cannot be empty")
        if not str(self.target).startswith("/"):
            raise ValueError("container mount target must be an absolute path")


@dataclass(frozen=True)
class ContainerOptions:
    """Options common to Docker and Apptainer/Singularity runtimes."""

    mounts: tuple[Mount, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    workdir: str = "/work"
    clean_env: bool = True
    gpus: int | str | None = None
    ports: tuple[tuple[int, int], ...] = ()
    network: str | None = None
    extra_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.workdir.startswith("/"):
            raise ValueError("container workdir must be an absolute path")
        object.__setattr__(self, "mounts", tuple(self.mounts))
        object.__setattr__(self, "extra_args", tuple(self.extra_args))
        object.__setattr__(self, "env", dict(self.env))
        ports = tuple((int(host), int(container)) for host, container in self.ports)
        if any(host < 1 or container < 1 for host, container in ports):
            raise ValueError("container ports must be positive")
        object.__setattr__(self, "ports", ports)


@dataclass(frozen=True)
class SlurmOptions:
    """SLURM-specific submission options.

    ``extra_directives`` is intentionally available for site-specific flags;
    values are rendered as ``#SBATCH`` lines without invoking a shell locally.
    """

    job_name: str | None = None
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    nodes: int | str | None = None
    tasks: int | None = None
    array: str | None = None
    dependency: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    export: str | None = None
    prologue: tuple[str, ...] = ()
    epilogue: tuple[str, ...] = ()
    extra_directives: Mapping[str, str | int | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tasks is not None and self.tasks < 1:
            raise ValueError("tasks must be at least 1")
        object.__setattr__(self, "prologue", tuple(self.prologue))
        object.__setattr__(self, "epilogue", tuple(self.epilogue))
        object.__setattr__(self, "extra_directives", dict(self.extra_directives))
        for value in (self.job_name, self.partition, self.account, self.qos, self.stdout, self.stderr, self.export):
            if value is not None and any(char in str(value) for char in "\r\n\x00"):
                raise ValueError("SLURM options cannot contain control characters")


@dataclass(frozen=True)
class Interpreter:
    """An executable used to run a script, extensible beyond Bash and Python."""

    executable: tuple[str, ...]
    suffix: str = ".sh"

    def __init__(self, executable: str | Sequence[str], suffix: str = ".sh") -> None:
        if isinstance(executable, str):
            executable = (executable,)
        executable = tuple(executable)
        if not executable or not executable[0]:
            raise ValueError("interpreter executable cannot be empty")
        object.__setattr__(self, "executable", executable)
        object.__setattr__(self, "suffix", suffix if suffix.startswith(".") else f".{suffix}")

    @classmethod
    def bash(cls, executable: str = "bash") -> Interpreter:
        return cls(executable, ".sh")

    @classmethod
    def python(cls, executable: str = "python3") -> Interpreter:
        return cls(executable, ".py")


@dataclass(frozen=True)
class Command:
    argv: tuple[str, ...]

    def __init__(self, argv: Sequence[str]) -> None:
        argv = tuple(str(arg) for arg in argv)
        if not argv or not argv[0]:
            raise ValueError("command must contain an executable")
        object.__setattr__(self, "argv", argv)


@dataclass(frozen=True)
class Script:
    source: str | Path
    interpreter: Interpreter = field(default_factory=Interpreter.bash)
    args: tuple[str, ...] = ()
    inline: bool = False
    filename: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(str(arg) for arg in self.args))
        if self.inline and not isinstance(self.source, str):
            raise TypeError("inline script source must be text")

    @property
    def path(self) -> Path | None:
        if self.inline:
            return None
        return Path(self.source)

    def text(self) -> str:
        if self.inline:
            return str(self.source)
        return Path(self.source).read_text(encoding="utf-8")

    def name(self) -> str:
        if self.filename:
            return self.filename
        if not self.inline:
            return Path(self.source).name
        return f"job{self.interpreter.suffix}"


@dataclass(frozen=True)
class Job:
    """A backend-independent command or script specification."""

    payload: Command | Script
    name: str = "camri-job"
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: Path | str | None = None
    resources: Resources = field(default_factory=Resources)
    stdin: str | bytes | IO[str] | IO[bytes] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "env", dict(self.env))
        if not self.name or any(char in self.name for char in "\r\n\x00"):
            raise ValueError("job name cannot be empty")

    @classmethod
    def command(
        cls,
        argv: Sequence[str],
        *,
        name: str = "camri-job",
        env: Mapping[str, str] | None = None,
        cwd: Path | str | None = None,
        resources: Resources | None = None,
        stdin: str | bytes | IO[str] | IO[bytes] | None = None,
    ) -> Job:
        return cls(Command(argv), name, env or {}, cwd, resources or Resources(), stdin)

    @classmethod
    def script(
        cls,
        source: str | Path,
        *,
        interpreter: Interpreter | None = None,
        args: Sequence[str] = (),
        name: str = "camri-job",
        env: Mapping[str, str] | None = None,
        cwd: Path | str | None = None,
        resources: Resources | None = None,
        stdin: str | bytes | IO[str] | IO[bytes] | None = None,
    ) -> Job:
        return cls(
            Script(source, interpreter or Interpreter.bash(), tuple(args)),
            name,
            env or {},
            cwd,
            resources or Resources(),
            stdin,
        )

    @classmethod
    def inline(
        cls,
        source: str,
        *,
        interpreter: Interpreter | None = None,
        args: Sequence[str] = (),
        name: str = "camri-job",
        env: Mapping[str, str] | None = None,
        cwd: Path | str | None = None,
        resources: Resources | None = None,
        stdin: str | bytes | IO[str] | IO[bytes] | None = None,
    ) -> Job:
        return cls(
            Script(source, interpreter or Interpreter.bash(), tuple(args), inline=True),
            name,
            env or {},
            cwd,
            resources or Resources(),
            stdin,
        )


@dataclass(frozen=True)
class JobResult:
    job_id: str
    state: JobState
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    started_at: float | None = None
    ended_at: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.state is JobState.SUCCEEDED and self.returncode == 0


class JobHandle:
    """Common lifecycle interface returned by every worker."""

    job_id: str

    def status(self) -> JobState:
        raise NotImplementedError

    def wait(self, timeout: float | None = None, poll_interval: float = 2.0) -> JobResult:
        raise NotImplementedError

    def cancel(self) -> None:
        raise NotImplementedError

    def logs(self, stream: str = "stdout", follow: bool = False) -> str:
        raise NotImplementedError

    def result(self) -> JobResult | None:
        raise NotImplementedError

    def cleanup(self) -> None:
        raise NotImplementedError


__all__ = [
    "Command",
    "ContainerOptions",
    "Interpreter",
    "Job",
    "JobHandle",
    "JobResult",
    "JobState",
    "Mount",
    "Resources",
    "Script",
    "SlurmOptions",
]
