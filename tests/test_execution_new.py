from pathlib import Path

from camri.execution import (
    ContainerOptions,
    DockerRuntime,
    DockerWorker,
    Interpreter,
    Job,
    LocalWorker,
    Mount,
    SingularityWorker,
    SlurmOptions,
    SlurmWorker,
)


def _executable(path: Path, text: str) -> str:
    path.write_text(text)
    path.chmod(0o755)
    return str(path)


def test_local_inline_script_and_handle(tmp_path):
    worker = LocalWorker(artifact_root=tmp_path / "jobs")
    handle = worker.submit(Job.inline("printf 'ok'", interpreter=Interpreter.bash(), name="echo"))
    result = handle.wait(timeout=5)
    assert result.success
    assert result.stdout == "ok"
    assert handle.status().value == "SUCCEEDED"
    handle.cleanup()
    assert not handle.artifact_dir.exists()


def test_custom_interpreter_and_argv_are_rendered_without_shell(tmp_path):
    worker = LocalWorker(artifact_root=tmp_path / "jobs")
    job = Job.inline("printf '%s' \"$1\"", interpreter=Interpreter("sh"), args=["a b"])
    result = worker.run(job, timeout=5, check=True)
    assert result.stdout == "a b"


def test_command_stdin_is_forwarded(tmp_path):
    worker = LocalWorker(artifact_root=tmp_path / "jobs")
    result = worker.run(Job.command(["cat"], stdin="input"), timeout=5, check=True)
    assert result.stdout == "input"


def test_docker_runtime_builds_mounts_and_detach(tmp_path):
    runtime = DockerRuntime(
        "image:test",
        ContainerOptions(mounts=(Mount(tmp_path, "/data", read_only=True),)),
        executable="/bin/echo",
    )
    job = Job.command(["echo", "hello"], name="docker job")
    command = runtime.build_command(job, tmp_path / "artifact", tmp_path, detach=True)
    assert command[:3] == ["/bin/echo", "run", "--detach"]
    assert "--volume" in command
    assert "image:test" in command


def test_docker_worker_handle_lifecycle_with_fake_cli(tmp_path):
    docker = _executable(
        tmp_path / "docker",
        """#!/bin/sh
case "$1" in
run) printf 'cid123\\n' ;;
inspect) case "$*" in *Status*) printf 'exited\\n' ;; *) printf '0\\n' ;; esac ;;
logs) printf 'container-output\\n' ;;
stop|rm) exit 0 ;;
esac
""",
    )
    worker = DockerWorker("image:test", artifact_root=tmp_path / "jobs", executable=docker)
    handle = worker.submit(Job.command(["echo", "inside"]))
    assert handle.wait(timeout=1).success
    assert handle.logs() == "container-output\n"
    handle.cleanup()


def test_singularity_worker_uses_same_job_api_with_fake_cli(tmp_path):
    apptainer = _executable(tmp_path / "apptainer", "#!/bin/sh\nprintf 'singularity-output\\n'\n")
    worker = SingularityWorker("image.sif", artifact_root=tmp_path / "jobs", executable=apptainer)
    result = worker.run(Job.command(["echo", "inside"]), timeout=5, check=True)
    assert result.stdout == "singularity-output\n"


def test_slurm_submission_accepts_commands_unavailable_on_login_node(tmp_path):
    captured = tmp_path / "script.sh"
    sbatch = _executable(tmp_path / "sbatch", f"#!/bin/sh\ncat > '{captured}'\nprintf '123;cluster\\n'\n")
    squeue = _executable(tmp_path / "squeue", "#!/bin/sh\nexit 0\n")
    sacct = _executable(tmp_path / "sacct", "#!/bin/sh\ncase \"$*\" in *format=ExitCode*) printf '0:0\\n' ;; *) printf 'COMPLETED|0:0\\n' ;; esac\n")
    scancel = _executable(tmp_path / "scancel", "#!/bin/sh\nexit 0\n")
    worker = SlurmWorker(
        options=SlurmOptions(partition="base", account="lab"),
        artifact_root=tmp_path / "jobs",
        executable=sbatch,
        squeue=squeue,
        sacct=sacct,
        scancel=scancel,
    )
    handle = worker.submit(
        Job.command(["site-specific-command", "--flag"], name="cluster", env={"CAMRI_TEST": "a b"}),
        options=SlurmOptions(partition="debug"),
    )
    assert handle.job_id == "123"
    script = captured.read_text()
    assert "#SBATCH --partition=debug" in script
    assert "#SBATCH --account=lab" in script
    assert "export CAMRI_TEST='a b'" in script
    assert "site-specific-command --flag" in script
    assert handle.wait(timeout=1).success
