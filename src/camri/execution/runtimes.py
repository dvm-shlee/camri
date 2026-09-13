"""Command rendering for host and container runtimes."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Protocol

from .errors import WorkerUnavailableError
from .models import Command, ContainerOptions, Job, Mount, Script


class Runtime(Protocol):
    def build_command(
        self,
        job: Job,
        artifact_dir: Path,
        host_cwd: Path,
        *,
        detach: bool = False,
        validate: bool = True,
    ) -> list[str]: ...


def _require_executable(executable: str) -> None:
    if os.path.sep in executable:
        available = Path(executable).is_file()
    else:
        available = shutil.which(executable) is not None
    if not available:
        raise WorkerUnavailableError(f"executable not found: {executable}")


def _script_path(job: Job, artifact_dir: Path, container_path: str | None = None, source_base: Path | None = None) -> Path | str:
    payload = job.payload
    if not isinstance(payload, Script):
        raise TypeError("job does not contain a script")
    filename = payload.name()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", filename):
        filename = "script" + payload.interpreter.suffix
    artifact_dir.mkdir(parents=True, exist_ok=True)
    target = artifact_dir / filename
    source = Path(payload.source)
    if not payload.inline and not source.is_absolute() and source_base is not None:
        source = source_base / source
    text = payload.text() if payload.inline else source.read_text(encoding="utf-8")
    target.write_text(text, encoding="utf-8")
    target.chmod(0o700)
    return f"{container_path.rstrip('/')}/{filename}" if container_path else target


class NativeRuntime:
    """Run a command directly on the host."""

    def build_command(self, job: Job, artifact_dir: Path, host_cwd: Path, *, detach: bool = False, validate: bool = True) -> list[str]:
        if isinstance(job.payload, Command):
            command = list(job.payload.argv)
        else:
            script = _script_path(job, artifact_dir, None, host_cwd)
            command = [*job.payload.interpreter.executable, str(script), *job.payload.args]
        if validate:
            _require_executable(command[0])
        return command


class DockerRuntime:
    """Render a Docker CLI invocation."""

    def __init__(self, image: str, options: ContainerOptions | None = None, executable: str = "docker") -> None:
        if not image:
            raise ValueError("Docker image cannot be empty")
        self.image = image
        self.options = options or ContainerOptions()
        self.executable = executable

    def build_command(self, job: Job, artifact_dir: Path, host_cwd: Path, *, detach: bool = False, validate: bool = True) -> list[str]:
        if validate:
            _require_executable(self.executable)
        opts = self.options
        command = [self.executable, "run"]
        if detach:
            command.append("--detach")
        command.extend(["--name", _safe_container_name(job.name, artifact_dir.name)])
        command.extend(["--workdir", opts.workdir])
        command.extend(["--volume", f"{host_cwd.resolve()}:{opts.workdir}:rw"])
        command.extend(["--volume", f"{artifact_dir.resolve()}:/camri_job:rw"])
        for mount in opts.mounts:
            command.extend(["--volume", _docker_mount(mount, host_cwd)])
        for key, value in {**opts.env, **job.env}.items():
            command.extend(["--env", f"{key}={value}"])
        if opts.network:
            command.extend(["--network", opts.network])
        for host_port, container_port in opts.ports:
            command.extend(["--publish", f"{host_port}:{container_port}"])
        if opts.gpus is not None:
            command.extend(["--gpus", str(opts.gpus)])
        if job.resources.cpus:
            command.extend(["--cpus", str(job.resources.cpus)])
        if job.resources.memory:
            command.extend(["--memory", job.resources.memory])
        if job.resources.gpus is not None and opts.gpus is None:
            command.extend(["--gpus", str(job.resources.gpus)])
        command.extend(opts.extra_args)
        command.append(self.image)
        if isinstance(job.payload, Command):
            command.extend(job.payload.argv)
        else:
            script = _script_path(job, artifact_dir, "/camri_job", host_cwd)
            command.extend([*job.payload.interpreter.executable, str(script), *job.payload.args])
        return command


class SingularityRuntime:
    """Render a Singularity or Apptainer ``exec`` invocation."""

    def __init__(self, image: str | Path, options: ContainerOptions | None = None, executable: str | None = None) -> None:
        if not str(image):
            raise ValueError("Singularity image cannot be empty")
        self.image = str(image)
        self.options = options or ContainerOptions()
        self.executable = executable

    def _executable(self, *, validate: bool = True) -> str:
        if self.executable:
            if validate:
                _require_executable(self.executable)
            return self.executable
        for candidate in ("apptainer", "singularity"):
            if shutil.which(candidate):
                return candidate
        if not validate:
            return "apptainer"
        raise WorkerUnavailableError("neither apptainer nor singularity was found")

    def build_command(self, job: Job, artifact_dir: Path, host_cwd: Path, *, detach: bool = False, validate: bool = True) -> list[str]:
        del detach  # Apptainer itself has no daemon detach mode.
        opts = self.options
        command = [self._executable(validate=validate), "exec"]
        if opts.clean_env:
            command.append("--cleanenv")
        command.extend(["--pwd", opts.workdir])
        command.extend(["--bind", f"{host_cwd.resolve()}:{opts.workdir}"])
        command.extend(["--bind", f"{artifact_dir.resolve()}:/camri_job"])
        for mount in opts.mounts:
            command.extend(["--bind", _singularity_mount(mount, host_cwd)])
        for key, value in {**opts.env, **job.env}.items():
            command.extend(["--env", f"{key}={value}"])
        if opts.gpus is not None or job.resources.gpus is not None:
            command.append("--nv")
        command.extend(opts.extra_args)
        command.append(self.image)
        if isinstance(job.payload, Command):
            command.extend(job.payload.argv)
        else:
            script = _script_path(job, artifact_dir, "/camri_job", host_cwd)
            command.extend([*job.payload.interpreter.executable, str(script), *job.payload.args])
        return command


def _safe_container_name(name: str, key: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "camri-job"
    return f"{cleaned[:40]}-{key}".replace(" ", "-")


def _mount_source(mount: Mount, base: Path) -> Path:
    source = Path(mount.source).expanduser()
    return (base / source).resolve() if not source.is_absolute() else source.resolve()


def _docker_mount(mount: Mount, base: Path) -> str:
    suffix = ":ro" if mount.read_only else ":rw"
    return f"{_mount_source(mount, base)}:{mount.target}{suffix}"


def _singularity_mount(mount: Mount, base: Path) -> str:
    suffix = ":ro" if mount.read_only else ""
    return f"{_mount_source(mount, base)}:{mount.target}{suffix}"


__all__ = ["DockerRuntime", "NativeRuntime", "Runtime", "SingularityRuntime"]
