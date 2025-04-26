import numpy.ma as ma
from .base import BasePlotter
from ..plane import Plane
from ..selector import ImageSelector

class OverlayPlotter(BasePlotter):
    """
    OverlayPlotter overlays additional data on top of an existing figure's axes
    (e.g. those created by an OrthoPlotter) without modifying the main image frame.
    
    This class takes an ImageSelector as input and is specialized for adding an overlay
    on top of the already plotted main image. The overlay is displayed via new imshow objects
    with a high z-order, so that xlim/ylim and other axis settings remain unchanged.
    
    Features include:
      - Threshold masking (values outside the specified (lower, upper) range are masked).
      - Customizable colormap, alpha transparency, and interpolation for the overlay.
      - Methods to update the overlay display and to remove the overlay.
    
    Parameters:
        selector (ImageSelector): The image selector for the main image.
        coords (tuple[float, float, float], optional): The slice coordinates (default: center).
        coordinate_system (str, optional): 'world' or 'index' (default: 'world').
        threshold (tuple[float, float], optional): (lower, upper) threshold for masking.
            Values outside this range are masked.
        cmap (str or Colormap, optional): Colormap for the overlay (default: 'hot').
        alpha (float, optional): Transparency for the overlay (default: 0.5).
        interpolation (str, optional): Interpolation method (default: 'nearest').
    """
    def __init__(self, 
                 selector: ImageSelector, 
                 coords: tuple[float, float, float] = None, 
                 coordinate_system: str = 'world',
                 threshold: tuple[float, float] = None,
                 cmap='hot',
                 alpha: float = 0.5,
                 interpolation: str = 'nearest'):
        
        # Call the BasePlotter initializer with the provided parameters.
        super().__init__(selector=selector, 
                         coords=coords, 
                         coordinate_system=coordinate_system)
        self.threshold = threshold
        self.cmap = cmap
        self.alpha = alpha
        self.interpolation = interpolation
        self.axesimages = {}  # Dictionary to store overlay AxesImage objects per plane
        self._init_slices()

    def _init_slices(self):
        """
        Initialize overlay slices and their extents for each orthogonal plane.
        
        This method assumes that the ImageSelector.get_slice method accepts an 'overlay'
        flag to return the overlay data and its extent.
        """
        self.slices = {}
        self.extents = {}
        for p in Plane:
            # The selector is expected to return the overlay slice if overlay=True.
            slice_img, extent = self.selector.get_slice(p, self.coords[p.value], 
                                                         self.coordinate_system, overlay=True)
            self.slices[p] = slice_img
            self.extents[p] = extent

    def plot(self, fig=None):
        """
        Overlay the data on the existing figure's axes.
        
        The overlay is added on top of the main image (with a high z-order) without modifying 
        the axis limits (xlim/ylim). This method assumes that the figure already has exactly 3 axes.
        """
        if fig is None:
            if hasattr(self, 'fig'):
                fig = self.fig
            else:
                raise ValueError("OverlayPlotter.plot(): No figure available.")
        self.fig = fig
        axes = fig.get_axes()
        if len(axes) != 3:
            raise ValueError("Orthogonal view requires exactly 3 axes.")
        self.axes = {}
        for p in Plane:
            ax = axes[p.value]
            self.axes[p] = ax
            slice_img = self.slices[p]
            # Apply threshold masking if specified
            if self.threshold is not None:
                lower, upper = self.threshold
                slice_img = ma.masked_outside(slice_img, lower, upper)
            # Add the overlay image with a high zorder so it appears above the main image.
            im = ax.imshow(slice_img, extent=self.extents[p], 
                           cmap=self.cmap, alpha=self.alpha, 
                           interpolation=self.interpolation, zorder=10)
            self.axesimages[p] = im
        self.fig.canvas.draw_idle()

    def update_overlay(self, threshold=None, cmap=None, alpha=None):
        """
        Update the overlay parameters and refresh the overlay display.
        
        Parameters:
            threshold (tuple[float, float], optional): New (lower, upper) threshold for masking.
            cmap (str or Colormap, optional): New colormap for the overlay.
            alpha (float, optional): New transparency value for the overlay.
        """
        if threshold is not None:
            self.threshold = threshold
        if cmap is not None:
            self.cmap = cmap
        if alpha is not None:
            self.alpha = alpha
        
        # Recalculate overlay slices if needed
        self._init_slices()
        
        # Update each overlay artist without modifying the axes
        for p in Plane:
            if p in self.axesimages:
                slice_img = self.slices[p]
                if self.threshold is not None:
                    lower, upper = self.threshold
                    slice_img = ma.masked_outside(slice_img, lower, upper)
                self.axesimages[p].set_data(slice_img)
                self.axesimages[p].set_cmap(self.cmap)
                self.axesimages[p].set_alpha(self.alpha)
                self.axesimages[p].set_interpolation(self.interpolation)
            else:
                ax = self.axes[p]
                slice_img = self.slices[p]
                if self.threshold is not None:
                    lower, upper = self.threshold
                    slice_img = ma.masked_outside(slice_img, lower, upper)
                overlay_im = ax.imshow(slice_img, extent=self.extents[p],
                                       cmap=self.cmap, alpha=self.alpha, 
                                       interpolation=self.interpolation, zorder=10)
                self.axesimages[p] = overlay_im
        self.fig.canvas.draw_idle()

    def remove_overlay(self):
        """
        Remove the overlay from all axes.
        """
        for artist in self.axesimages.values():
            artist.remove()
        self.axesimages = {}
        self.fig.canvas.draw_idle()
