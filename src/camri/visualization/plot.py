"""Matplotlib plotting primitives; notebook integrations live in adapters."""

from __future__ import annotations

import numpy as np

from camri.image import as_canonical, world_to_voxel


def _volume(image, volume: int = 0) -> tuple[object, np.ndarray]:
    image = as_canonical(image)
    if image.ndim == 4:
        if volume < 0 or volume >= image.shape[-1]:
            raise IndexError("volume is outside the image")
        data = np.asarray(image.dataobj[..., volume], dtype=float)
    elif image.ndim == 3:
        data = np.asarray(image.dataobj, dtype=float)
    else:
        raise ValueError("image must be 3D or 4D")
    return image, data


def _overlay_data(overlay, image, volume: int):
    overlay = as_canonical(overlay)
    if overlay.shape[:3] != image.shape[:3] or not np.allclose(overlay.affine, image.affine, atol=1e-5):
        raise ValueError("overlay and image must have the same spatial grid")
    return np.asarray(overlay.dataobj[..., volume] if overlay.ndim == 4 else overlay.dataobj, dtype=float)


def plot_slice(image, *, plane: str = "axial", index: int | None = None, volume: int = 0, ax=None, clim=None, cmap="gray", overlay=None, overlay_cmap="magma", overlay_alpha: float = 0.5):
    import matplotlib.pyplot as plt

    image, data = _volume(image, volume)
    planes = {"sagittal": 0, "coronal": 1, "axial": 2}
    try:
        axis = planes[plane.lower()]
    except KeyError as exc:
        raise ValueError("plane must be sagittal, coronal, or axial") from exc
    if index is None:
        index = data.shape[axis] // 2
    if index < 0 or index >= data.shape[axis]:
        raise IndexError("slice index is outside the image")
    if ax is None:
        _, ax = plt.subplots()
    slicer: list[slice | int] = [slice(None)] * 3
    slicer[axis] = index
    shown = np.asarray(data[tuple(slicer)]).T
    ax.imshow(np.rot90(shown), cmap=cmap, clim=clim, origin="lower")
    if overlay is not None:
        ov = _overlay_data(overlay, image, volume)
        ax.imshow(np.rot90(ov[tuple(slicer)]), cmap=overlay_cmap, alpha=overlay_alpha, origin="lower")
    ax.set_title(f"{plane} [{index}]")
    ax.set_axis_off()
    return ax


def plot_ortho(image, *, coords=None, volume: int = 0, axes=None, clim=None, cmap="gray", overlay=None, overlay_cmap="magma", overlay_alpha: float = 0.5):
    import matplotlib.pyplot as plt

    image, data = _volume(image, volume)
    if coords is None:
        voxel = np.asarray(data.shape) / 2.0
    else:
        voxel = np.asarray(world_to_voxel(image, coords, rounding="nearest"), dtype=int)
    if voxel.shape != (3,) or np.any(voxel < 0) or np.any(voxel >= np.asarray(data.shape)):
        raise ValueError("coords must be a valid world-coordinate triple")
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, plane, idx in zip(axes, ("sagittal", "coronal", "axial"), voxel):
        plot_slice(image, plane=plane, index=int(idx), volume=volume, ax=ax, clim=clim, cmap=cmap, overlay=overlay, overlay_cmap=overlay_cmap, overlay_alpha=overlay_alpha)
    return axes


def plot_mosaic(image, *, volume: int = 0, n_slices: int = 12, axes=None, clim=None, cmap="gray", overlay=None, overlay_cmap="magma", overlay_alpha: float = 0.5):
    import matplotlib.pyplot as plt

    image, data = _volume(image, volume)
    if n_slices < 1:
        raise ValueError("n_slices must be positive")
    indices = np.linspace(0, data.shape[2] - 1, n_slices, dtype=int)
    ncols = int(np.ceil(np.sqrt(len(indices))))
    nrows = int(np.ceil(len(indices) / ncols))
    if axes is None:
        _, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    axes = np.asarray(axes).reshape(-1)
    for ax, index in zip(axes, indices):
        plot_slice(image, plane="axial", index=int(index), volume=volume, ax=ax, clim=clim, cmap=cmap, overlay=overlay, overlay_cmap=overlay_cmap, overlay_alpha=overlay_alpha)
    for ax in axes[len(indices):]:
        ax.set_visible(False)
    return axes


__all__ = ["plot_mosaic", "plot_ortho", "plot_slice"]
