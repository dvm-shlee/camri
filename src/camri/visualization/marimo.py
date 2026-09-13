"""Optional marimo widget adapter."""

from __future__ import annotations


class MarimoViewer:
    def __init__(self, image, overlay=None) -> None:
        self.image = image
        self.overlay = overlay

    def show(self, *, volume: int = 0, coords=None, clim=None):
        try:
            import marimo as mo
        except ImportError as exc:
            raise ImportError("install camri[marimo] to use MarimoViewer") from exc
        import matplotlib.pyplot as plt

        from camri.image import load_image

        from .plot import plot_ortho

        image = load_image(self.image)
        max_volume = max(0, image.shape[-1] - 1) if getattr(image, "ndim", 3) == 4 else 0
        slider = mo.ui.slider(0, max_volume, value=volume, label="volume")
        figure, axes = plt.subplots(1, 3, figsize=(12, 4))
        plot_ortho(image, volume=slider.value, coords=coords, axes=axes, clim=clim, overlay=self.overlay)
        return mo.vstack([slider, figure])


__all__ = ["MarimoViewer"]
