"""Basic resting-state functional connectivity metrics."""

from __future__ import annotations

from itertools import product

import numpy as np
from scipy import signal, stats
from scipy.stats import rankdata


def _signals(data, time_axis: int = -1) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("connectivity data must have shape (signals, time)")
    axis = int(time_axis)
    if axis < 0:
        axis += arr.ndim
    if axis < 0 or axis >= arr.ndim:
        raise ValueError("time_axis is out of range")
    arr = np.moveaxis(arr, axis, -1)
    if not np.isfinite(arr).all():
        raise ValueError("data contains NaN or Inf")
    if arr.shape[-1] < 2:
        raise ValueError("at least two time points are required")
    return arr


def _zscore(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = data.mean(axis=-1, keepdims=True)
    std = data.std(axis=-1, keepdims=True)
    return np.divide(data - mean, std, out=np.full_like(data, np.nan), where=std > 0), std[..., 0]


def correlation_matrix(data, *, time_axis: int = -1) -> np.ndarray:
    arr = _signals(data, time_axis)
    z, std = _zscore(arr)
    result = z @ z.T / arr.shape[-1]
    invalid = std == 0
    result[invalid, :] = np.nan
    result[:, invalid] = np.nan
    valid = ~invalid
    result[np.ix_(valid, valid)] = np.clip(result[np.ix_(valid, valid)], -1.0, 1.0)
    result[np.diag_indices_from(result)] = np.where(valid, 1.0, np.nan)
    return result


def seed_correlation(data, seed, *, time_axis: int = -1) -> np.ndarray:
    arr = _signals(data, time_axis)
    seed = np.asarray(seed, dtype=np.float64).reshape(-1)
    if seed.shape[0] != arr.shape[-1]:
        raise ValueError("seed length must match the number of time points")
    values, seed_std = _zscore(seed[None, :])
    z, std = _zscore(arr)
    result = (z @ values[0]) / arr.shape[-1]
    result[std == 0] = np.nan
    if seed_std[0] == 0:
        result[:] = np.nan
    return np.clip(result, -1.0, 1.0)


def correlation_to_t(r, n_timepoints: int) -> tuple[np.ndarray, np.ndarray]:
    if n_timepoints <= 2:
        raise ValueError("at least three time points are required")
    values = np.asarray(r, dtype=np.float64)
    clipped = np.clip(values, -1.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_value = clipped * np.sqrt(n_timepoints - 2) / np.sqrt(1.0 - clipped**2)
    p_value = 2.0 * stats.t.sf(np.abs(t_value), n_timepoints - 2)
    perfect = np.isfinite(clipped) & (np.abs(clipped) == 1)
    t_value = np.where(perfect, np.sign(clipped) * np.inf, t_value)
    p_value = np.where(perfect, 0.0, p_value)
    return t_value, p_value


def connectivity_strength(data, *, pval: float | None = None, positive: bool = False, absolute: bool = False) -> np.ndarray:
    matrix = correlation_matrix(data)
    np.fill_diagonal(matrix, np.nan)
    if pval is not None:
        if not 0 < pval <= 1:
            raise ValueError("pval must be in (0, 1]")
        _, p_values = correlation_to_t(matrix, np.asarray(data).shape[-1])
        matrix[p_values >= pval] = np.nan
    if positive:
        matrix[matrix < 0] = 0
    if absolute:
        matrix = np.abs(matrix)
    return np.nansum(matrix, axis=-1)


def alff(
    data,
    tr: float,
    *,
    lowcut: float = 0.01,
    highcut: float = 0.1,
    fraction: bool = False,
    time_axis: int = -1,
    nperseg: int | None = None,
) -> np.ndarray:
    arr = _signals(data, time_axis)
    nyquist = 0.5 / tr if tr > 0 else 0
    if tr <= 0 or lowcut < 0 or highcut <= lowcut or highcut >= nyquist:
        raise ValueError("invalid tr or frequency band")
    if nperseg is None:
        nperseg = min(256, arr.shape[-1])
    elif nperseg < 1:
        raise ValueError("nperseg must be positive")
    frequencies, power = signal.welch(arr, fs=1.0 / tr, axis=-1, nperseg=nperseg, scaling="density")
    positive = frequencies > 0
    band = positive & (frequencies >= lowcut) & (frequencies <= highcut)
    if not band.any():
        raise ValueError("frequency band contains no Welch bins")
    amplitude = np.sqrt(power)
    value = amplitude[..., band].sum(axis=-1)
    if fraction:
        denominator = amplitude[..., positive].sum(axis=-1)
        value = np.divide(value, denominator, out=np.full_like(value, np.nan), where=denominator > 0)
    return value


def falff(data, tr: float, **kwargs) -> np.ndarray:
    kwargs["fraction"] = True
    return alff(data, tr, **kwargs)


def _tie_correction(ranks: np.ndarray) -> float:
    correction = 0.0
    for row in ranks:
        _, counts = np.unique(row, return_counts=True)
        correction += float(np.sum(counts**3 - counts))
    return correction


def kendall_w(data: np.ndarray) -> float:
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Kendall data must have shape (observers, items)")
    observers, items = values.shape
    if observers < 2 or items < 2:
        return np.nan
    ranks = rankdata(values, axis=1, method="average")
    rank_sum = ranks.sum(axis=0)
    s = np.sum((rank_sum - rank_sum.mean()) ** 2)
    denominator = observers**2 * (items**3 - items) - observers * _tie_correction(ranks)
    return float(12.0 * s / denominator) if denominator > 0 else np.nan


def reho(data, *, mask=None, neighborhood: int = 26, time_axis: int = -1) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    axis = int(time_axis)
    if axis < 0:
        axis += arr.ndim
    if axis < 0 or axis >= arr.ndim:
        raise ValueError("time_axis is out of range")
    arr = np.moveaxis(arr, axis, -1)
    if arr.ndim != 4:
        raise ValueError("ReHo data must be 4D")
    if not np.isfinite(arr).all():
        raise ValueError("data contains NaN or Inf")
    if neighborhood not in {6, 18, 26}:
        raise ValueError("neighborhood must be 6, 18, or 26")
    mask_bool = np.ones(arr.shape[:3], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask_bool.shape != arr.shape[:3]:
        raise ValueError("mask must match the spatial image shape")
    offsets = []
    for offset in product((-1, 0, 1), repeat=3):
        if offset == (0, 0, 0):
            continue
        distance = sum(abs(v) for v in offset)
        max_distance = max(abs(v) for v in offset)
        if neighborhood == 6 and distance == 1 or neighborhood == 18 and max_distance <= 1 and distance <= 2 or neighborhood == 26:
            offsets.append(offset)
    result = np.full(arr.shape[:3], np.nan, dtype=np.float64)
    for x, y, z in zip(*np.nonzero(mask_bool)):
        points = [(x, y, z)]
        for dx, dy, dz in offsets:
            point = (x + dx, y + dy, z + dz)
            if all(0 <= point[i] < arr.shape[i] for i in range(3)) and mask_bool[point]:
                points.append(point)
        result[x, y, z] = kendall_w(arr[tuple(np.asarray(points).T)])
    return result


def _image_signals(image, mask):
    from camri.image import load_image

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
    return image, data, mask_bool


def alff_image(image, tr: float, *, mask=None, **kwargs):
    import nibabel as nib

    from camri.image import unmask

    image, data, mask_bool = _image_signals(image, mask)
    values = alff(data[mask_bool], tr, **kwargs)
    mask_image = nib.Nifti1Image(mask_bool.astype(np.uint8), image.affine)
    return unmask(values, mask_image, template=image)


def reho_image(image, *, mask=None, neighborhood: int = 26):
    import nibabel as nib


    image, data, mask_bool = _image_signals(image, mask)
    header = image.header.copy()
    header.set_data_shape(data.shape[:3])
    header.set_data_dtype(np.float32)
    return nib.Nifti1Image(reho(data, mask=mask_bool, neighborhood=neighborhood).astype(np.float32), image.affine, header=header)


__all__ = [
    "alff",
    "alff_image",
    "connectivity_strength",
    "correlation_matrix",
    "correlation_to_t",
    "falff",
    "kendall_w",
    "reho",
    "reho_image",
    "seed_correlation",
]
