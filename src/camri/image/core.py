"""Small, metadata-preserving NIfTI helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy.ndimage import gaussian_filter

ImageLike = str | Path | nib.spatialimages.SpatialImage


def load_image(image: ImageLike) -> nib.spatialimages.SpatialImage:
    if isinstance(image, (str, Path)):
        path = Path(image).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        return cast(nib.spatialimages.SpatialImage, nib.load(str(path)))
    if isinstance(image, nib.spatialimages.SpatialImage):
        return image
    raise TypeError("image must be a path or nibabel SpatialImage")


def save_image(image: ImageLike, path: str | Path) -> Path:
    image = load_image(image)
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(target))
    return target


def _new_like(template: nib.spatialimages.SpatialImage, data: np.ndarray, affine: np.ndarray | None = None, dtype=None):
    data = np.asarray(data, dtype=dtype)
    header = template.header.copy()
    header.set_data_shape(data.shape)
    if dtype is not None:
        header.set_data_dtype(data.dtype)
    return template.__class__(data, template.affine.copy() if affine is None else np.asarray(affine), header=header)


def as_canonical(image: ImageLike) -> nib.spatialimages.SpatialImage:
    return nib.as_closest_canonical(load_image(image))


def crop_image(image: ImageLike, voxel_min: Iterable[int], voxel_max: Iterable[int]):
    image = load_image(image)
    lo = tuple(int(v) for v in voxel_min)
    hi = tuple(int(v) for v in voxel_max)
    if len(lo) != 3 or len(hi) != 3:
        raise ValueError("voxel bounds must contain three values")
    if any(v < 0 for v in lo) or any(b <= a for a, b in zip(lo, hi)):
        raise ValueError("invalid crop bounds")
    if any(b > s for b, s in zip(hi, image.shape[:3])):
        raise ValueError("crop bounds exceed image shape")
    slices = tuple(slice(a, b) for a, b in zip(lo, hi)) + (Ellipsis,)
    data = np.asarray(image.dataobj[slices])
    affine = image.affine.copy()
    affine[:3, 3] = nib.affines.apply_affine(image.affine, lo)
    return _new_like(image, data, affine)


def resample_to(image: ImageLike, target: ImageLike, order: int = 1):
    image, target = load_image(image), load_image(target)
    if order not in range(6):
        raise ValueError("order must be between 0 and 5")
    return resample_from_to(image, (target.shape, target.affine), order=order)


def resample_voxels(image: ImageLike, voxel_sizes: Iterable[float], order: int = 1):
    image = load_image(image)
    sizes = np.asarray(tuple(float(v) for v in voxel_sizes))
    if sizes.shape != (3,) or np.any(~np.isfinite(sizes)) or np.any(sizes <= 0):
        raise ValueError("voxel_sizes must contain three positive finite values")
    old_sizes = np.linalg.norm(image.affine[:3, :3], axis=0)
    directions = image.affine[:3, :3] / old_sizes
    shape = tuple(np.maximum(1, np.ceil(np.asarray(image.shape[:3]) * old_sizes / sizes).astype(int)))
    affine = np.eye(4)
    affine[:3, :3] = directions * sizes
    affine[:3, 3] = image.affine[:3, 3]
    return resample_from_to(image, (shape + image.shape[3:], affine), order=order)


def _check_grid(image, mask) -> None:
    if mask.ndim != 3:
        raise ValueError("mask must be a 3D image")
    if image.shape[:3] != mask.shape[:3] or not np.allclose(image.affine, mask.affine, atol=1e-5):
        raise ValueError("image and mask must have the same spatial grid")


def apply_mask(image: ImageLike, mask: ImageLike):
    image, mask = load_image(image), load_image(mask)
    _check_grid(image, mask)
    mask_bool = np.asarray(mask.dataobj) != 0
    data = np.asarray(image.get_fdata()).copy()
    if data.ndim == 3:
        data[~mask_bool] = 0
    elif data.ndim == 4:
        data[~mask_bool, :] = 0
    else:
        raise ValueError("image must be 3D or 4D")
    return _new_like(image, data, dtype=np.float32)


def unmask(data: np.ndarray, mask: ImageLike, template: ImageLike | None = None):
    mask = load_image(mask)
    if mask.ndim != 3:
        raise ValueError("mask must be a 3D image")
    mask_bool = np.asarray(mask.dataobj) != 0
    values = np.asarray(data)
    n_voxels = int(mask_bool.sum())
    if values.ndim == 1 and values.shape[0] != n_voxels:
        raise ValueError("masked data length does not match mask")
    if values.ndim == 2 and values.shape[0] != n_voxels:
        raise ValueError("masked data first axis does not match mask")
    out_shape = mask.shape[:3] + (() if values.ndim == 1 else (values.shape[1],))
    out = np.zeros(out_shape, dtype=values.dtype)
    out[mask_bool] = values
    return _new_like(load_image(template) if template is not None else mask, out, dtype=values.dtype)


def smooth_image(image: ImageLike, fwhm: float, mask: ImageLike | None = None):
    image = load_image(image)
    if not np.isfinite(fwhm) or fwhm <= 0:
        raise ValueError("fwhm must be positive and finite")
    sigma = float(fwhm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    voxel_sizes = np.linalg.norm(image.affine[:3, :3], axis=0)
    sigma_vox = sigma / voxel_sizes
    mask_bool = None
    if mask is not None:
        mask = load_image(mask)
        _check_grid(image, mask)
        mask_bool = np.asarray(mask.dataobj) != 0
    data = np.asarray(image.get_fdata(), dtype=np.float64)

    def smooth_3d(volume):
        if mask_bool is None:
            return gaussian_filter(volume, sigma_vox, mode="nearest")
        weights = gaussian_filter(mask_bool.astype(float), sigma_vox, mode="constant", cval=0)
        smoothed = gaussian_filter(volume * mask_bool, sigma_vox, mode="constant", cval=0)
        result = np.divide(smoothed, weights, out=np.zeros_like(smoothed), where=weights > np.finfo(float).eps)
        result[~mask_bool] = 0
        return result

    if data.ndim == 3:
        result = smooth_3d(data)
    elif data.ndim == 4:
        result = np.stack([smooth_3d(data[..., i]) for i in range(data.shape[-1])], axis=-1)
    else:
        raise ValueError("image must be 3D or 4D")
    return _new_like(image, result, dtype=np.float32)


def trim_volumes(image: ImageLike, start: int = 0, end: int | None = None, step: int = 1):
    image = load_image(image)
    if image.ndim < 4:
        return image
    if step == 0:
        raise ValueError("step cannot be zero")
    data = np.asarray(image.dataobj[..., slice(start, end, step)])
    return _new_like(image, data)


def voxel_to_world(image: ImageLike, indices):
    image = load_image(image)
    values = np.asarray(indices, dtype=float)
    if values.shape[-1] != 3:
        raise ValueError("indices must end in three coordinates")
    return nib.affines.apply_affine(image.affine, values)


def world_to_voxel(image: ImageLike, coordinates, *, rounding: str | None = None):
    image = load_image(image)
    values = np.asarray(coordinates, dtype=float)
    if values.shape[-1] != 3:
        raise ValueError("coordinates must end in three values")
    indices = nib.affines.apply_affine(np.linalg.inv(image.affine), values)
    if rounding == "nearest":
        return np.rint(indices).astype(int)
    if rounding == "floor":
        return np.floor(indices).astype(int)
    if rounding is not None:
        raise ValueError("rounding must be None, 'nearest', or 'floor'")
    return indices


__all__ = [
    "ImageLike",
    "apply_mask",
    "as_canonical",
    "crop_image",
    "load_image",
    "resample_to",
    "resample_voxels",
    "save_image",
    "smooth_image",
    "trim_volumes",
    "unmask",
    "voxel_to_world",
    "world_to_voxel",
]
