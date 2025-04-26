import numpy as np
from ..selector import ImageSelector


class BasePlotter:
    """
    Base class for plotting NIfTI images.
    For use as base class to implement common feature to MosaicPlotter and SlicePlotter.
    This class provides a common interface and shared functionality for plotting NIfTI images.
    It is not intended to be used directly.
    """

    def __init__(self, 
                 selector: ImageSelector, 
                 coords: tuple[float,float,float] = None, 
                 coordinate_system='world',
                 clim=None,
                 contrast_percentile: tuple[float, float] = (None, None),
                 interpolation='nearest',
                 cmap=None,
                 hide_spines=False,
                 figsize=None,
                 dpi=None):
        self.selector = selector
        self.set_coords(coords, coordinate_system)
        self._clim = clim
        self.contrast_percentile = contrast_percentile
        self.cmap = cmap or 'gray'
        self.interpolation = interpolation
        self.hide_spines = hide_spines 
        self.figsize = figsize or (12, 4)
        self.dpi = dpi or 100

    def plot(self):
        """
        Plot the NIfTI image.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def update(self):
        """
        update the NIfTI image.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _init_slices(self):
        """
        Initialize the slices for the plot.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _update_slices(self):
        """
        Update the slices for the plot.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _init_viewports(self):
        """
        Initialize the viewports for the plot.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def img(self):
        return self.selector.img.get_nibobj()

    @property
    def clim(self):
        if self._clim is None:
            self._clim = (self.vmin, self.vmax)
        return self._clim
    
    @property
    def vmin(self):
        if self.contrast_percentile[0] is not None:
            return np.quantile(self.img.dataobj, self.contrast_percentile[0])
        return self.img.dataobj.min()
    
    @property
    def vmax(self):
        if self.contrast_percentile[1] is not None:
            return np.quantile(self.img.dataobj, self.contrast_percentile[1])
        return self.img.dataobj.max()

    def set_coords(self, coords, coordinate_system='world'):
        """
        Set the coordinates for the orthogonal planes.
        """
        self._previous_coords = getattr(self, 'coords', None)
        self._previous_coordinate_system = getattr(self, 'coordinate_system', None)

        if self._previous_coordinate_system != coordinate_system:
            self.coordinate_system = coordinate_system

        if coords is None:
            if not hasattr(self, 'coords'):
                world_center = self.selector.get_world_center()
                self.coords = (
                    world_center if coordinate_system == 'world' 
                    else self.selector.img.coord_to_index(world_center)
                )
        else:
            self.coords = coords