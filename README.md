# CAMRI

CAMRI is a small Python toolbox for neuroimaging work. It provides one job
specification that can run on the host, in Docker, in Singularity/Apptainer, or
through a local SLURM CLI. The same package also contains metadata-preserving
NIfTI helpers, signal preprocessing, QC metrics, basic rsFC metrics, BIDS
queries, and optional Jupyter or marimo viewers.

CAMRI requires Python 3.11 or newer. The base installation contains only NumPy,
SciPy, and NiBabel:

```bash
python -m pip install camri
python -m pip install 'camri[jupyter]'  # notebook viewer
python -m pip install 'camri[marimo]'   # marimo viewer
python -m pip install 'camri[bids]'     # full PyBIDS adapter
```

Development uses the repository-local Python environment managed by uv:

```bash
uv sync
uv run pytest
uv run ruff check src tests
uv run mypy src/camri
```

## Execute the same job in different environments

```python
from camri.execution import (
    ContainerOptions, DockerWorker, Interpreter, Job, Mount,
    SingularityRuntime, SingularityWorker, SlurmOptions, SlurmWorker,
)

job = Job.script(
    "analysis.py",
    interpreter=Interpreter.python(),
    args=["--subject", "sub-01"],
    name="camri-analysis",
)

local = DockerWorker(
    "nipreps/fmriprep:latest",
    options=ContainerOptions(mounts=(Mount("/data", "/data", read_only=True),)),
)
handle = local.submit(job)
result = handle.wait()

cluster = SlurmWorker(
    options=SlurmOptions(partition="general", account="lab"),
)
cluster_result = cluster.run(job, check=True)

singularity = SingularityWorker("/shared/containers/tool.sif")
singularity_result = singularity.run(job, check=True)

cluster_container = SlurmWorker(
    options=SlurmOptions(partition="general"),
    runtime=SingularityRuntime("/shared/containers/tool.sif"),
)
cluster_container_result = cluster_container.run(job, check=True)
```

Commands use argv safely. Shell syntax is available through an explicit
`Job.inline(..., interpreter=Interpreter.bash())` or a user supplied
`Interpreter`, so R, Julia and site-specific launchers do not require a new
worker class.

## Image and QC examples

```python
import numpy as np

from camri.image import load_image, smooth_image
from camri.qc import dvars, tsnr
from camri.connectivity import correlation_matrix

image = load_image("sub-01_task-rest_bold.nii.gz")
smoothed = smooth_image(image, fwhm=6.0)
mask = np.ones(image.shape[:3], dtype=bool)
signals = image.get_fdata()[mask != 0]
quality = {"tSNR": tsnr(signals), "DVARS": dvars(signals)}
connectivity = correlation_matrix(signals)
```

Jupyter and marimo integrations are optional. The base package can be used for
array processing, NIfTI transformations, BIDS queries, and job submission
without installing notebook dependencies.

## 0.1 migration

The 0.1 release intentionally uses new namespaces. Replace
`camri.manager.SlurmWorker` with `camri.execution.SlurmWorker`, move signal
functions from `camri.prep` to `camri.processing`, QC functions to `camri.qc`,
and rsFC functions to `camri.connectivity`. NIfTI I/O is in `camri.image` and
plots/viewers are in `camri.visualization`. The unfinished `camri.stats`
package is removed until a separate statistics adapter is designed.
