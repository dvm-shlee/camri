import matplotlib.pyplot as plt
from .base import BasePlotter
from ..plane import Plane
from ..selector import ImageSelector
from ..plotter import SlicePlotter
from copy import deepcopy as copy


class OrthoPlotter(BasePlotter):
    """
    Class to plot 3D NIfTI images in orthogonal planes.
    """
    def __init__(self, 
                 selector: ImageSelector, 
                 coords: tuple[float,float,float] = None, 
                 coordinate_system='world',
                 volume_id=None,
                 clim=None,
                 contrast_percentile: tuple[float, float] = (None, None),
                 interpolation='nearest',
                 cmap=None,
                 crosslines=True,
                 crosslines_kwargs=None,
                 hide_spines=False,
                 figsize=None,
                 dpi=None):
        
        super().__init__(selector=selector, 
                         coords=coords, 
                         coordinate_system=coordinate_system,
                         clim=clim, 
                         contrast_percentile=contrast_percentile, 
                         interpolation=interpolation, 
                         cmap=cmap, 
                         hide_spines=hide_spines, 
                         figsize=figsize, 
                         dpi=dpi)
        self._init_slices(volume_id)
        self._init_viewports()
        self.crosslines = crosslines
        self.crosslines_kwargs = crosslines_kwargs or {}

    def _init_slices(self, volume_id):
        """
        Update the slices for the orthogonal planes.
        """
        self.slices = {}
        self.extents = {}
        self.set_volume_id(volume_id)

    def _init_viewports(self):
        """
        Set axis limits so that the displayed viewport is square and centered.

        Parameters
        ----------
        ax : plt.Axes
            The matplotlib axes to modify
        plane : str
            Anatomical plane ('axial', 'coronal', or 'sagittal')
        """
        self._viewports = {}
        for p in Plane:
            x_idx, y_idx = Plane.get_others(p)
            center_x = self.selector.affine[x_idx, 3] + self.selector.voxel_sizes[x_idx] * self.selector.shape[x_idx]/2
            center_y = self.selector.affine[y_idx, 3] + self.selector.voxel_sizes[y_idx] * self.selector.shape[y_idx]/2
            vp = self.selector.viewport_size
            self._viewports[p] = (center_x - vp/2, center_x + vp/2, center_y - vp/2, center_y + vp/2)

    def _update_slices(self):
        for p in Plane:
            if self.slices.get(p) is not None:
                if (self._previous_coords[p.value] == self.coords[p.value] and
                    self._previous_coordinate_system == self.coordinate_system and
                    self._previous_volume_id == self.volume_id):
                    # skip if slice coordinate has not changed
                    continue
            slice_img, extent = self.selector.get_slice(p, self.coords[p.value], self.coordinate_system, volume_id=self.volume_id)
            self.slices[p] = slice_img
            self.extents[p] = extent

    def set_volume_id(self, volume_id):
        """
        Set the volume ID for 4D images.

        If the input volume_id is None, keep the existing volume_id if available, otherwise initialize to 0.
        On the first assignment, _previous_volume_id is set to -1 to force an update.
        """
        current = getattr(self, 'volume_id', None)
        
        if current is None:
            # 첫 설정: 입력이 None이면 0, 아니면 주어진 값을 사용
            self.volume_id = volume_id if volume_id is not None else 0
            self._previous_volume_id = -1
        else:
            # 입력이 None이면 기존 값 유지
            new_volume = volume_id if volume_id is not None else current
            self._previous_volume_id = self.volume_id
            if self._previous_volume_id != new_volume:
                self.volume_id = new_volume
        self._update_slices()


    def plot(self, fig=None):
        if fig is None:
            if hasattr(self, 'fig'):
                fig = self.fig
                axes = fig.get_axes()
            else:
                fig, axes = plt.subplots(1, 3, figsize=self.figsize, dpi=self.dpi)
                self.fig = fig
        else:
            axes = fig.get_axes()
            self.fig = fig
            
        if len(axes) != 3:
            raise ValueError("Figure must have exactly 3 axes for orthogonal planes.")
        
        self.axes = {}
        for p in Plane:
            ax = axes[p.value]
            slice_img = self.slices[p]
            extent = self.extents[p]
            clim = self.clim
            cmap = self.cmap
            interpolation = self.interpolation
            xmin, xmax, ymin, ymax = self._viewports[p]
            self.axes[p] = SlicePlotter(ax=ax, 
                                        data=slice_img, 
                                        extent=extent, 
                                        clim=clim,
                                        interpolation=interpolation, 
                                        cmap=cmap)
            self.axes[p].set_xlim(xmin, xmax)
            self.axes[p].set_ylim(ymin, ymax)
            if self.crosslines:
                x_idx, y_idx = Plane.get_others(p)
                coords = self.coords if self.coordinate_system == 'world' else self.selector.img.index_to_coord(*self.coords)
                self.axes[p].add_crosslines(coords[x_idx], coords[y_idx], **self.crosslines_kwargs)
            if self.hide_spines:
                ax.axis('off')

    def update(self, coords=None, volume_id=None, coordinate_system='world'):
        """
        Update the slices and viewports.
        """
        self.set_coords(coords, coordinate_system)
        self.set_volume_id(volume_id)
        for p in Plane:
            # Update the slice and extent
            slice_img = self.slices[p]
            extent = self.extents[p]
            clim = self.clim
            cmap = self.cmap
            xmin, xmax, ymin, ymax = self._viewports[p]
            self.axes[p].update_data(slice_img)
            self.axes[p].update_extent(extent)
            self.axes[p].set_contrast(clim[0], clim[1])
            self.axes[p].set_cmap(cmap)
            self.axes[p].set_xlim(xmin, xmax)
            self.axes[p].set_ylim(ymin, ymax)
            if self.crosslines:
                x_idx, y_idx = Plane.get_others(p)
                coords = self.coords if self.coordinate_system == 'world' else self.selector.img.index_to_coord(*self.coords)
                self.axes[p].update_crosslines(coords[x_idx], coords[y_idx], **self.crosslines_kwargs)
            else:
                self.axes[p].remove_crosslines()
            if self.hide_spines:
                self.axes[p].im.axes.axis('off')
            self.axes[p].draw()
