"""Optional Jupyter widget adapter."""

from __future__ import annotations

from dataclasses import dataclass

from camri.image import ImageLike


@dataclass
class JupyterViewer:
    image: ImageLike
    overlay: ImageLike | None = None

    def show(self, *, volume: int = 0, coords=None, clim=None):
        try:
            import matplotlib.pyplot as plt
            from IPython.display import display
            from ipywidgets import IntSlider, Output, VBox
        except ImportError as exc:
            raise ImportError("install camri[jupyter] to use JupyterViewer") from exc
        from camri.image import load_image

        from .plot import plot_ortho

        output = Output()
        image = load_image(self.image)
        max_volume = max(0, getattr(image, "shape", (0, 0, 0, 1))[-1] - 1) if getattr(image, "ndim", 3) == 4 else 0
        slider = IntSlider(value=volume, min=0, max=max_volume, description="volume")

        def redraw(_=None):
            with output:
                output.clear_output(wait=True)
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                plot_ortho(image, volume=slider.value, coords=coords, axes=axes, clim=clim, overlay=self.overlay)
                display(fig)
                plt.close(fig)

        slider.observe(redraw, names="value")
        redraw()
        box = VBox([output, slider])
        display(box)
        return box


__all__ = ["JupyterViewer"]
