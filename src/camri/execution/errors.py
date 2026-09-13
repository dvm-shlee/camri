class ExecutionError(RuntimeError):
    """Base exception for execution failures."""


class WorkerUnavailableError(ExecutionError):
    """The requested backend or runtime executable is unavailable."""


class SubmissionError(ExecutionError):
    """A backend rejected a job submission."""


class JobExecutionError(ExecutionError):
    """A synchronous job completed unsuccessfully."""

    def __init__(self, result):
        self.result = result
        super().__init__(f"job {result.job_id} ended in {result.state.value} (exit={result.returncode})")


class JobTimeoutError(TimeoutError, ExecutionError):
    """Waiting for a job exceeded the requested timeout."""


__all__ = [
    "ExecutionError",
    "JobExecutionError",
    "JobTimeoutError",
    "SubmissionError",
    "WorkerUnavailableError",
]
