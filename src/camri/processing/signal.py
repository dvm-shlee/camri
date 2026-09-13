"""Validated signal preprocessing primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


def _array(data, time_axis: int = -1) -> tuple[np.ndarray, int]:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 0:
        raise ValueError("data must have at least one dimension")
    axis = int(time_axis)
    if axis < 0:
        axis += arr.ndim
    if axis < 0 or axis >= arr.ndim:
        raise ValueError("time_axis is out of range")
    if not np.isfinite(arr).all():
        raise ValueError("data contains NaN or Inf")
    return np.moveaxis(arr, axis, -1), axis


def standardize(data, *, time_axis: int = -1, ddof: int = 0):
    arr, axis = _array(data, time_axis)
    if ddof < 0 or ddof >= arr.shape[-1]:
        raise ValueError("ddof must be non-negative and smaller than the number of time points")
    mean = arr.mean(axis=-1, keepdims=True)
    std = arr.std(axis=-1, ddof=ddof, keepdims=True)
    result = np.divide(arr - mean, std, out=np.full_like(arr, np.nan), where=std > 0)
    return np.moveaxis(result, -1, axis)


def temporal_filter(
    data,
    tr: float,
    *,
    lowcut: float | None = None,
    highcut: float | None = None,
    order: int = 4,
    time_axis: int = -1,
):
    arr, axis = _array(data, time_axis)
    if not np.isfinite(tr) or tr <= 0:
        raise ValueError("tr must be positive and finite")
    if order < 1:
        raise ValueError("order must be positive")
    nyquist = 0.5 / tr
    if lowcut is None and highcut is None:
        raise ValueError("at least one of lowcut or highcut is required")
    if lowcut is not None and (lowcut <= 0 or lowcut >= nyquist):
        raise ValueError("lowcut must be between 0 and the Nyquist frequency")
    if highcut is not None and (highcut <= 0 or highcut >= nyquist):
        raise ValueError("highcut must be between 0 and the Nyquist frequency")
    if lowcut is not None and highcut is not None and lowcut >= highcut:
        raise ValueError("lowcut must be smaller than highcut")
    if lowcut is not None and highcut is not None:
        cutoff: float | list[float] = [lowcut / nyquist, highcut / nyquist]
        btype = "bandpass"
    elif lowcut is not None:
        cutoff = lowcut / nyquist
        btype = "highpass"
    else:
        assert highcut is not None
        cutoff = highcut / nyquist
        btype = "lowpass"
    sos = signal.butter(order, cutoff, btype=btype, output="sos")
    try:
        result = signal.sosfiltfilt(sos, arr, axis=-1)
    except ValueError as exc:
        raise ValueError("time series is too short for the requested filter") from exc
    return np.moveaxis(result, -1, axis)


def build_polynomial_confounds(n_timepoints: int, order: int = 1, *, include_intercept: bool = True) -> np.ndarray:
    if n_timepoints < 1 or order < 0:
        raise ValueError("n_timepoints must be positive and order must be non-negative")
    x = np.linspace(-1.0, 1.0, n_timepoints)
    start = 0 if include_intercept else 1
    return np.column_stack([x**degree for degree in range(start, order + 1)])


@dataclass(frozen=True)
class RegressionResult:
    cleaned: np.ndarray
    betas: np.ndarray
    design: np.ndarray
    rank: int


def regress_confounds(
    data,
    confounds,
    *,
    time_axis: int = -1,
    add_intercept: bool = True,
) -> RegressionResult:
    arr, axis = _array(data, time_axis)
    design = np.asarray(confounds, dtype=np.float64)
    if design.ndim == 1:
        design = design[:, None]
    if design.ndim != 2 or design.shape[0] != arr.shape[-1]:
        raise ValueError("confounds must have shape (time, regressors)")
    if not np.isfinite(design).all():
        raise ValueError("confounds contain NaN or Inf")
    if add_intercept:
        design = np.column_stack([np.ones(design.shape[0]), design])
    flat = arr.reshape(-1, arr.shape[-1]).T
    betas, _, rank, _ = np.linalg.lstsq(design, flat, rcond=None)
    cleaned = (flat - design @ betas).T.reshape(arr.shape)
    return RegressionResult(np.moveaxis(cleaned, -1, axis), betas, design, int(rank))


def trim_volumes(data, start: int = 0, end: int | None = None, step: int = 1, *, time_axis: int = -1):
    arr = np.asarray(data)
    axis = int(time_axis)
    if axis < 0:
        axis += arr.ndim
    if axis < 0 or axis >= arr.ndim:
        raise ValueError("time_axis is out of range")
    if step == 0:
        raise ValueError("step cannot be zero")
    index = [slice(None)] * arr.ndim
    index[axis] = slice(start, end, step)
    return arr[tuple(index)]


__all__ = [
    "RegressionResult",
    "build_polynomial_confounds",
    "regress_confounds",
    "standardize",
    "temporal_filter",
    "trim_volumes",
]
