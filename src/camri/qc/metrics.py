"""Quality-control metrics with explicit time-axis and invalid-value rules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _time_last(data, time_axis: int = -1) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim < 1:
        raise ValueError("data must have at least one dimension")
    axis = int(time_axis)
    if axis < 0:
        axis += arr.ndim
    if axis < 0 or axis >= arr.ndim:
        raise ValueError("time_axis is out of range")
    if not np.isfinite(arr).all():
        raise ValueError("data contains NaN or Inf")
    if arr.shape[-1] < 2:
        raise ValueError("at least two time points are required")
    return np.moveaxis(arr, axis, -1)


def tsnr(data, *, time_axis: int = -1, ddof: int = 0) -> np.ndarray:
    arr = _time_last(data, time_axis)
    if arr.shape[-1] <= ddof:
        raise ValueError("not enough time points for ddof")
    mean = arr.mean(axis=-1)
    std = arr.std(axis=-1, ddof=ddof)
    return np.divide(mean, std, out=np.full_like(mean, np.nan), where=std > 0)


def dvars(data, *, mask=None, time_axis: int = -1) -> np.ndarray:
    arr = _time_last(data, time_axis)
    flat = arr.reshape(-1, arr.shape[-1])
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != arr.shape[:-1]:
            raise ValueError("mask must match all non-time dimensions")
        flat = flat[mask.reshape(-1)]
    if flat.shape[0] == 0:
        raise ValueError("mask selects no voxels")
    return np.sqrt(np.mean(np.diff(flat, axis=-1) ** 2, axis=0))


def framewise_displacement(
    motion,
    *,
    radius_mm: float = 50.0,
    rotation_unit: str = "radians",
) -> np.ndarray:
    values = motion_to_mm(motion, radius_mm=radius_mm, rotation_unit=rotation_unit)
    displacement = np.abs(np.diff(values, axis=0, prepend=values[:1]))
    return displacement.sum(axis=1)


def motion_to_mm(motion, *, radius_mm: float = 50.0, rotation_unit: str = "radians") -> np.ndarray:
    """Convert rotational motion parameters to arc-distance millimetres."""
    values = np.asarray(motion, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("motion must have shape (time, 6): 3 rotations followed by 3 translations")
    if not np.isfinite(values).all() or not np.isfinite(radius_mm) or radius_mm <= 0:
        raise ValueError("motion must be finite and radius_mm must be positive")
    values = values.copy()
    if rotation_unit == "degrees":
        values[:, :3] = np.deg2rad(values[:, :3])
    elif rotation_unit != "radians":
        raise ValueError("rotation_unit must be 'radians' or 'degrees'")
    values[:, :3] *= radius_mm
    return values


def read_motion(path: str | Path, *, radius_mm: float | None = None, rotation_unit: str = "degrees") -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] != 6:
        raise ValueError("motion file must contain six columns")
    if radius_mm is not None:
        return motion_to_mm(values, radius_mm=radius_mm, rotation_unit=rotation_unit)
    return values


def tsnr_image(image, *, mask=None, time_axis: int = -1, ddof: int = 0):
    """Compute a voxelwise tSNR map and preserve the input image grid."""
    import nibabel as nib

    from camri.image import load_image, unmask

    image = load_image(image)
    data = np.asarray(image.get_fdata(), dtype=np.float64)
    if data.ndim != 4:
        raise ValueError("image must be 4D")
    if mask is None:
        mask_bool = np.ones(data.shape[:3], dtype=bool)
    else:
        mask_img = load_image(mask)
        if mask_img.ndim != 3 or mask_img.shape[:3] != data.shape[:3] or not np.allclose(mask_img.affine, image.affine, atol=1e-5):
            raise ValueError("image and mask must have the same spatial grid")
        mask_bool = np.asarray(mask_img.dataobj) != 0
    if not mask_bool.any():
        raise ValueError("mask selects no voxels")
    values = tsnr(data[mask_bool], time_axis=-1, ddof=ddof)
    mask_image = nib.Nifti1Image(mask_bool.astype(np.uint8), image.affine)
    return unmask(values, mask_image, template=image)


def dvars_image(image, *, mask=None, time_axis: int = -1) -> np.ndarray:
    """Compute the temporal DVARS series from a 4D image."""
    from camri.image import load_image

    image = load_image(image)
    data = np.asarray(image.get_fdata(), dtype=np.float64)
    if data.ndim != 4:
        raise ValueError("image must be 4D")
    mask_bool = None
    if mask is not None:
        mask_img = load_image(mask)
        if mask_img.ndim != 3 or mask_img.shape[:3] != data.shape[:3] or not np.allclose(mask_img.affine, image.affine, atol=1e-5):
            raise ValueError("image and mask must have the same spatial grid")
        mask_bool = np.asarray(mask_img.dataobj) != 0
    return dvars(data, mask=mask_bool, time_axis=time_axis)


@dataclass(frozen=True)
class BoldSummary:
    mean: np.ndarray
    std: np.ndarray
    tsnr: np.ndarray
    dvars: np.ndarray


def bold_summary(data, *, mask=None, time_axis: int = -1, ddof: int = 0) -> BoldSummary:
    arr = _time_last(data, time_axis)
    if ddof < 0 or ddof >= arr.shape[-1]:
        raise ValueError("ddof must be non-negative and smaller than the number of time points")
    flat = arr.reshape(-1, arr.shape[-1])
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != arr.shape[:-1]:
            raise ValueError("mask must match all non-time dimensions")
        flat = flat[mask.reshape(-1)]
    if flat.shape[0] == 0:
        raise ValueError("mask selects no voxels")
    mean = flat.mean(axis=0)
    std = flat.std(axis=0, ddof=ddof)
    return BoldSummary(mean, std, np.divide(mean, std, out=np.full_like(mean, np.nan), where=std > 0), dvars(flat))


__all__ = ["BoldSummary", "bold_summary", "dvars", "dvars_image", "framewise_displacement", "motion_to_mm", "read_motion", "tsnr", "tsnr_image"]
