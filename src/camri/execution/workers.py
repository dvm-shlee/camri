"""Local, Docker, Singularity and SLURM job workers."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import uuid
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from .errors import (
    JobExecutionError,
    JobTimeoutError,
    SubmissionError,
    WorkerUnavailableError,
)
from .models import (
    ContainerOptions,
    Job,
    JobHandle,
    JobResult,
    JobState,
    Resources,
    SlurmOptions,
)
from .runtimes import DockerRuntime, NativeRuntime, Runtime, SingularityRuntime


def _now() -> float:
    return time.time()


def _merge_env(job: Job, runtime_env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    if runtime_env:
        env.update({str(k): str(v) for k, v in runtime_env.items()})
    env.update({str(k): str(v) for k, v in job.env.items()})
    return env


def _artifact_root(root: Path | str | None) -> Path:
    path = Path(root) if root is not None else Path.cwd() / ".camri" / "jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    except OSError:
        return ""


class _BaseHandle(JobHandle):
    def __init__(self, job_id: str, artifact_dir: Path, stdout_path: Path, stderr_path: Path) -> None:
        self.job_id = str(job_id)
        self.artifact_dir = artifact_dir
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self._result: JobResult | None = None
        self._cancelled = False

    def logs(self, stream: str = "stdout", follow: bool = False) -> str:
        del follow  # A non-blocking snapshot is safer in notebooks.
        if stream not in {"stdout", "stderr"}:
            raise ValueError("stream must be 'stdout' or 'stderr'")
        return _read(self.stdout_path if stream == "stdout" else self.stderr_path)

    def result(self) -> JobResult | None:
        return self._result

    def cleanup(self) -> None:
        if self.status() in {JobState.PENDING, JobState.RUNNING}:
            raise RuntimeError(f"cannot clean up active job {self.job_id}")
        if self.artifact_dir.exists():
            shutil.rmtree(self.artifact_dir)


class LocalJobHandle(_BaseHandle):
    def __init__(self, process: subprocess.Popen, *args, started_at: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.process = process
        self.started_at = started_at

    def status(self) -> JobState:
        if self._cancelled:
            return JobState.CANCELLED
        code = self.process.poll()
        if code is None:
            return JobState.RUNNING
        return JobState.SUCCEEDED if code == 0 else JobState.FAILED

    def wait(self, timeout: float | None = None, poll_interval: float = 2.0) -> JobResult:
        del poll_interval
        if self._result is not None:
            return self._result
        try:
            code = self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise JobTimeoutError(f"timed out waiting for job {self.job_id}") from exc
        ended = _now()
        state = JobState.CANCELLED if self._cancelled else (JobState.SUCCEEDED if code == 0 else JobState.FAILED)
        self._result = JobResult(
            self.job_id,
            state,
            code,
            self.logs("stdout"),
            self.logs("stderr"),
            self.started_at,
            ended,
            {"pid": self.process.pid},
        )
        return self._result

    def cancel(self) -> None:
        if self.process.poll() is None:
            self._cancelled = True
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


class DockerJobHandle(_BaseHandle):
    def __init__(self, worker: DockerWorker, container_id: str, *args, started_at: float, **kwargs) -> None:
        super().__init__(container_id, *args, **kwargs)
        self.worker = worker
        self.started_at = started_at

    def status(self) -> JobState:
        if self._cancelled:
            return JobState.CANCELLED
        try:
            proc = subprocess.run(
                [self.worker.executable, "inspect", "--format", "{{.State.Status}}", self.job_id],
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError:
            return JobState.UNKNOWN
        if proc.returncode:
            return JobState.UNKNOWN
        state = proc.stdout.strip().lower()
        if state in {"created", "restarting"}:
            return JobState.PENDING
        if state == "running":
            return JobState.RUNNING
        if state == "exited":
            return JobState.SUCCEEDED if self._exit_code() == 0 else JobState.FAILED
        return JobState.UNKNOWN

    def _exit_code(self) -> int | None:
        proc = subprocess.run(
            [self.worker.executable, "inspect", "--format", "{{.State.ExitCode}}", self.job_id],
            text=True,
            capture_output=True,
            check=False,
        )
        try:
            return int(proc.stdout.strip())
        except (ValueError, TypeError):
            return None

    def wait(self, timeout: float | None = None, poll_interval: float = 2.0) -> JobResult:
        if self._result is not None:
            return self._result
        started_wait = _now()
        while True:
            state = self.status()
            if state in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}:
                code = self._exit_code()
                self._result = JobResult(self.job_id, state, code, self.logs("stdout"), self.logs("stderr"), self.started_at, _now())
                return self._result
            if timeout is not None and _now() - started_wait >= timeout:
                raise JobTimeoutError(f"timed out waiting for job {self.job_id}")
            time.sleep(poll_interval)

    def logs(self, stream: str = "stdout", follow: bool = False) -> str:
        if stream not in {"stdout", "stderr"}:
            raise ValueError("stream must be 'stdout' or 'stderr'")
        del follow
        proc = subprocess.run(
            [self.worker.executable, "logs", self.job_id],
            text=True,
            capture_output=True,
            check=False,
        )
        if stream == "stdout":
            return proc.stdout
        return proc.stderr

    def cancel(self) -> None:
        subprocess.run([self.worker.executable, "stop", self.job_id], capture_output=True, text=True, check=False)
        self._cancelled = True

    def cleanup(self) -> None:
        subprocess.run([self.worker.executable, "rm", "--force", self.job_id], capture_output=True, text=True, check=False)
        super().cleanup()


class SlurmJobHandle(_BaseHandle):
    def __init__(self, worker: SlurmWorker, job_id: str, *args, started_at: float, **kwargs) -> None:
        super().__init__(job_id, *args, **kwargs)
        self.worker = worker
        self.started_at = started_at

    def status(self) -> JobState:
        if self._cancelled:
            return JobState.CANCELLED
        return self.worker._status(self.job_id)

    def wait(self, timeout: float | None = None, poll_interval: float = 5.0) -> JobResult:
        if self._result is not None:
            return self._result
        started_wait = _now()
        while True:
            state = self.status()
            if state in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}:
                code = self.worker._exit_code(self.job_id)
                self._result = JobResult(self.job_id, state, code, self.logs("stdout"), self.logs("stderr"), self.started_at, _now())
                return self._result
            if timeout is not None and _now() - started_wait >= timeout:
                raise JobTimeoutError(f"timed out waiting for SLURM job {self.job_id}")
            time.sleep(poll_interval)

    def cancel(self) -> None:
        proc = subprocess.run([self.worker.scancel, self.job_id], text=True, capture_output=True, check=False)
        if proc.returncode:
            raise SubmissionError(proc.stderr.strip() or f"failed to cancel job {self.job_id}")
        self._cancelled = True


class Worker:
    """Common facade implemented by all CAMRI workers."""

    def submit(self, job: Job, **kwargs) -> JobHandle:
        raise NotImplementedError

    def run(self, job: Job, timeout: float | None = None, check: bool = False, **kwargs) -> JobResult:
        result = self.submit(job, **kwargs).wait(timeout=timeout)
        if check and not result.success:
            raise JobExecutionError(result)
        return result


class LocalWorker(Worker):
    def __init__(self, runtime: Runtime | None = None, artifact_root: Path | str | None = None) -> None:
        self.runtime = runtime or NativeRuntime()
        self.artifact_root = _artifact_root(artifact_root)

    def submit(self, job: Job, *, resources: Resources | None = None, **kwargs) -> LocalJobHandle:
        del resources, kwargs
        key = uuid.uuid4().hex[:12]
        artifact = self.artifact_root / key
        artifact.mkdir(parents=True)
        cwd = Path(job.cwd or Path.cwd()).expanduser().resolve()
        if not cwd.is_dir():
            raise FileNotFoundError(f"working directory does not exist: {cwd}")
        stdout = artifact / "stdout.log"
        stderr = artifact / "stderr.log"
        command = self.runtime.build_command(job, artifact, cwd)
        if job.stdin is None:
            stdin_data = None
        elif isinstance(job.stdin, (str, bytes)):
            stdin_data = job.stdin
        else:
            stdin_data = job.stdin.read()
        stdin = subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL
        text_mode = stdin_data is None or isinstance(stdin_data, str)
        try:
            with stdout.open("w", encoding="utf-8") as stdout_file, stderr.open("w", encoding="utf-8") as stderr_file:
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=_merge_env(job, getattr(getattr(self.runtime, "options", None), "env", None)),
                    stdin=stdin,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=text_mode,
                )
        except OSError as exc:
            raise WorkerUnavailableError(str(exc)) from exc
        if stdin_data is not None and process.stdin is not None:
            process.stdin.write(stdin_data)
            process.stdin.close()
        return LocalJobHandle(process, key, artifact, stdout, stderr, started_at=_now())


class DockerWorker(Worker):
    def __init__(self, image: str, options: ContainerOptions | None = None, artifact_root: Path | str | None = None, executable: str = "docker") -> None:
        self.runtime = DockerRuntime(image, options, executable)
        self.options = options or ContainerOptions()
        self.artifact_root = _artifact_root(artifact_root)
        self.executable = executable

    def submit(self, job: Job, *, resources: Resources | None = None, **kwargs) -> DockerJobHandle:
        del resources, kwargs
        key = uuid.uuid4().hex[:12]
        artifact = self.artifact_root / key
        artifact.mkdir(parents=True)
        cwd = Path(job.cwd or Path.cwd()).expanduser().resolve()
        stdout = artifact / "stdout.log"
        stderr = artifact / "stderr.log"
        command = self.runtime.build_command(job, artifact, cwd, detach=True)
        proc = subprocess.run(command, text=True, capture_output=True, check=False, cwd=cwd)
        if proc.returncode:
            raise SubmissionError(proc.stderr.strip() or proc.stdout.strip() or "docker run failed")
        container_id = proc.stdout.strip().splitlines()[-1].strip()
        if not container_id:
            raise SubmissionError("docker did not return a container id")
        return DockerJobHandle(self, container_id, artifact, stdout, stderr, started_at=_now())


class SingularityWorker(Worker):
    def __init__(self, image: str | Path, options: ContainerOptions | None = None, artifact_root: Path | str | None = None, executable: str | None = None) -> None:
        self.runtime = SingularityRuntime(image, options, executable)
        self.artifact_root = _artifact_root(artifact_root)

    def submit(self, job: Job, *, resources: Resources | None = None, **kwargs) -> LocalJobHandle:
        return LocalWorker(self.runtime, self.artifact_root).submit(job, resources=resources, **kwargs)


class SlurmWorker(Worker):
    def __init__(
        self,
        options: SlurmOptions | None = None,
        runtime: Runtime | None = None,
        artifact_root: Path | str | None = None,
        executable: str = "sbatch",
        squeue: str = "squeue",
        sacct: str = "sacct",
        scancel: str = "scancel",
    ) -> None:
        self.options = options or SlurmOptions()
        self.runtime = runtime or NativeRuntime()
        self.artifact_root = _artifact_root(artifact_root)
        self.executable, self.squeue, self.sacct, self.scancel = executable, squeue, sacct, scancel

    def submit(
        self,
        job: Job,
        *,
        options: SlurmOptions | None = None,
        resources: Resources | None = None,
        **kwargs,
    ) -> SlurmJobHandle:
        del kwargs
        key = uuid.uuid4().hex[:12]
        artifact = self.artifact_root / key
        artifact.mkdir(parents=True)
        cwd = Path(job.cwd or Path.cwd()).expanduser().resolve()
        if not cwd.is_dir():
            raise FileNotFoundError(f"working directory does not exist: {cwd}")
        effective_job = replace(job, resources=resources or job.resources)
        effective_options = _merge_slurm_options(self.options, options)
        stdout_template = _resolve_log_template(effective_options.stdout or str(artifact / "stdout-%j.log"), cwd)
        stderr_template = _resolve_log_template(effective_options.stderr or str(artifact / "stderr-%j.log"), cwd)
        script = self._render_script(effective_job, effective_options, artifact, cwd, stdout_template, stderr_template)
        (artifact / "batch.sh").write_text(script, encoding="utf-8")
        try:
            proc = subprocess.run(
                [self.executable, "--parsable"],
                input=script,
                text=True,
                capture_output=True,
                cwd=cwd,
                check=False,
            )
        except OSError as exc:
            shutil.rmtree(artifact, ignore_errors=True)
            raise WorkerUnavailableError(str(exc)) from exc
        if proc.returncode:
            shutil.rmtree(artifact, ignore_errors=True)
            raise SubmissionError(proc.stderr.strip() or proc.stdout.strip() or "sbatch failed")
        job_id = proc.stdout.strip().split(";")[0].splitlines()[-1].strip()
        if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id):
            shutil.rmtree(artifact, ignore_errors=True)
            raise SubmissionError(f"could not parse sbatch job id from: {proc.stdout!r}")
        stdout = Path(_slurm_substitute(stdout_template, job_id))
        stderr = Path(_slurm_substitute(stderr_template, job_id))
        return SlurmJobHandle(self, job_id, artifact, stdout, stderr, started_at=_now())

    def _render_script(self, job: Job, options: SlurmOptions, artifact: Path, cwd: Path, stdout: str, stderr: str) -> str:
        lines = ["#!/usr/bin/env bash"]
        directives: list[tuple[str, str | int | bool]] = [("job-name", options.job_name or job.name)]
        if job.resources.cpus:
            directives.append(("cpus-per-task", job.resources.cpus))
        if job.resources.memory:
            directives.append(("mem", job.resources.memory))
        if job.resources.time_limit:
            directives.append(("time", job.resources.time_limit))
        for key, value in (
            ("partition", options.partition),
            ("account", options.account),
            ("qos", options.qos),
            ("nodes", options.nodes),
            ("ntasks", options.tasks),
            ("array", options.array),
            ("dependency", options.dependency),
            ("export", options.export),
        ):
            if value is not None:
                directives.append((key, value))
        if job.resources.gpus is not None:
            directives.append(("gpus", job.resources.gpus))
        directives.extend((_directive_name(key), value) for key, value in options.extra_directives.items())
        directives.extend((("output", stdout), ("error", stderr)))
        for key, value in directives:
            if value is False:
                continue
            if value is True:
                lines.append(f"#SBATCH --{key}")
            else:
                lines.append(f"#SBATCH --{key}={value}")
        lines.extend(["set -euo pipefail", f"cd {shlex_quote(cwd.as_posix())}"])
        for key, value in job.env.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key)):
                raise ValueError(f"invalid environment variable name: {key}")
            lines.append(f"export {key}={shlex_quote(str(value))}")
        lines.extend(options.prologue)
        command = self.runtime.build_command(job, artifact, cwd, detach=False, validate=False)
        lines.append(" ".join(shlex_quote(arg) for arg in command))
        lines.extend(options.epilogue)
        return "\n".join(lines) + "\n"

    def _status(self, job_id: str) -> JobState:
        try:
            active = subprocess.run([self.squeue, "--noheader", "--format=%T", "--jobs", job_id], text=True, capture_output=True, check=False)
        except OSError:
            return JobState.UNKNOWN
        if active.returncode == 0 and active.stdout.strip():
            return _slurm_state(active.stdout.strip().splitlines()[0].strip())
        try:
            acct = subprocess.run([self.sacct, "-X", "-n", "-P", "-j", job_id, "--format=State,ExitCode"], text=True, capture_output=True, check=False)
        except OSError:
            return JobState.UNKNOWN
        if acct.returncode != 0 or not acct.stdout.strip():
            return JobState.UNKNOWN
        return _slurm_state(acct.stdout.strip().splitlines()[0].split("|")[0])

    def _exit_code(self, job_id: str) -> int | None:
        try:
            acct = subprocess.run([self.sacct, "-X", "-n", "-P", "-j", job_id, "--format=ExitCode"], text=True, capture_output=True, check=False)
            value = acct.stdout.strip().splitlines()[0].split(":")[0]
            return int(value)
        except (OSError, IndexError, ValueError):
            return None


def _slurm_state(value: str) -> JobState:
    value = value.strip().upper().split("+")[0]
    if value in {"PENDING", "CONFIGURING", "SUSPENDED"}:
        return JobState.PENDING
    if value in {"RUNNING", "COMPLETING"}:
        return JobState.RUNNING
    if value in {"COMPLETED"}:
        return JobState.SUCCEEDED
    if value in {"CANCELLED", "TIMEOUT", "PREEMPTED"}:
        return JobState.CANCELLED if value == "CANCELLED" else JobState.FAILED
    if value:
        return JobState.FAILED
    return JobState.UNKNOWN


def _slurm_substitute(path: str, job_id: str) -> str:
    base, _, task = job_id.partition("_")
    return path.replace("%j", job_id).replace("%A", base).replace("%a", task or "0")


def _directive_name(name: str) -> str:
    return str(name).lstrip("-")


def _resolve_log_template(template: str, cwd: Path) -> str:
    if any(char in template for char in "\r\n\x00"):
        raise ValueError("SLURM log path cannot contain control characters")
    path = Path(template).expanduser()
    return str(path if path.is_absolute() else cwd / path)


def _merge_slurm_options(base: SlurmOptions, override: SlurmOptions | None) -> SlurmOptions:
    if override is None:
        return base
    values = {}
    for field_name in SlurmOptions.__dataclass_fields__:
        value = getattr(override, field_name)
        if value is None or value == ():
            value = getattr(base, field_name)
        if field_name == "extra_directives":
            value = {**getattr(base, field_name), **getattr(override, field_name)}
        values[field_name] = value
    return SlurmOptions(**values)


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


__all__ = [
    "DockerWorker",
    "LocalWorker",
    "SingularityWorker",
    "SlurmWorker",
    "Worker",
]
