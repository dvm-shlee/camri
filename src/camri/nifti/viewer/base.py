import nibabel as nib
from .params import CrosslineParams
from ..plane import Plane
from ..handler import Nifti1Handler
from ..plotter import OrthoPlotter
from ..selector import ImageSelector
from ipywidgets import FloatSlider, IntSlider, HBox, VBox, Layout, Output
from IPython.display import display
import matplotlib.pyplot as plt
from collections import namedtuple


class BaseViewer:
    """
    A class for viewing NIfTI images.
    """
    def __init__(self, image: str | nib.Nifti1Image):
        """
        Initialize the NiftiViewer with a NIfTI image.

        Parameters
        ----------
        image : str | nib.Nifti0Image
            The path to the NIfTI image or a nibabel NIfTI image object.
        """
        self._set_containers()
        self.images['main'] = self._load_img(image)
        self.selector['main'] = ImageSelector(self.image)
        self._init_params()

    @staticmethod
    def _load_img(image: str | nib.Nifti1Image):
        if isinstance(image, str):
            return Nifti1Handler(nib.load(image))
        elif isinstance(image, nib.Nifti1Image):
            return Nifti1Handler(image)
        else:
            raise ValueError("Input must be a file path or a nibabel NIfTI image object.")

    @property
    def image(self):
        img = self.images['main']
        if img.is_oblique():
            return img.deoblique()
        else:
            return img

    @property
    def canonical_img(self):
        """
        Return the canonical image.
        """
        return self.selector['main'].img

    def _set_containers(self):
        self.fig = None
        self.cid = {}
        self.images = {'main': None}
        self.selector = {'main': None}
        self.plotter = {'main': None}
        self.outputs = {'ortho': Output(layout=Layout(border='0px solid gray'))}
        self.widgets = namedtuple('widgets', ['sliders', 'buttons'])
        self.layouts = {"full_center": Layout(width='100%', justify_content='center', align_items='center'),
                        "half_center": Layout(width='50%', justify_content='center', align_items='center'),}
        
        self.widgets.sliders = {"xyz": [],
                                "vol": None}
        self.widgets.buttons = {"crosslines": None,
                                "crossline_color": None,
                                "crossline_width": None,
                                "crossline_alpha": None,
                                "crossline_style": None,
                                "hide_spines": None}
        self.parameters = namedtuple('parameters', ['figsize',
                                                    'dpi',
                                                    'interpolation',
                                                    'cmap',
                                                    'contrast_percentile', 
                                                    'crosslines', 
                                                    'crosslines_kwargs', 
                                                    'hide_spines',
                                                    'coordinate_system'])

    def _init_params(self):
        self.set_coordinate_system()
        self.set_contrast_percentile()
        self.set_crosslines()
        self.set_hide_spines()
        self.set_cmap()
        self.set_figsize()
        self.set_interpolation()

    def set_figsize(self, figsize: tuple[float, float] = (12, 4), dpi:int = 100):
        self.parameters.figsize = figsize
        self.parameters.dpi = dpi

    def set_interpolation(self, interpolation: str = 'nearest'):
        """
        Set the interpolation method for the image.

        Parameters
        ----------
        interpolation : str
            The interpolation method to use ('nearest', 'bilinear', etc.).
        """
        if interpolation not in ['nearest', 'bilinear', 'bicubic']:
            raise ValueError("Interpolation must be 'nearest', 'bilinear', or 'bicubic'.")
        self.parameters.interpolation = interpolation

    def set_coordinate_system(self, coordinate_system: str = 'world'):
        """
        Set the coordinate system for the image.

        Parameters
        ----------
        coordinate_system : str
            The coordinate system to use ('world' or 'voxel').
        """
        if coordinate_system not in ['world', 'index']:
            raise ValueError("Coordinate system must be 'world' or 'index'.")
        self.parameters.coordinate_system = coordinate_system
    
    def set_contrast_percentile(self, lower: float = 0.001, higher: float = 0.998):
        """
        Set the contrast percentile for the image.

        Parameters
        ----------
        contrast_percentile : tuple[float, float]
            The lower and upper percentiles for contrast adjustment.
        """
        self.parameters.contrast_percentile = (lower, higher)

    def set_crosslines(self, crosslines: bool = True, crosslines_kwargs: CrosslineParams = None):
        """
        Set whether to display crosslines on the orthogonal views.

        Parameters
        ----------
        crosslines : bool
            Whether to display crosslines.
        crosslines_kwargs : dict
            Additional arguments for customizing the crosslines.
        """
        self.parameters.crosslines = crosslines
        self.parameters.crosslines_kwargs = crosslines_kwargs or CrosslineParams()
        
    def set_hide_spines(self, hide_spines: bool = False):
        """
        Set whether to hide the spines on the orthogonal views.

        Parameters
        ----------
        hide_spines : bool
            Whether to hide the spines.
        """
        self.parameters.hide_spines = hide_spines

    def set_cmap(self, cmap: str = 'gray'):
        """
        Set the colormap for the image.

        Parameters
        ----------
        cmap : str
            The colormap to use for the image.
        """
        self.parameters.cmap = cmap

    def init_widgets(self, 
                     coords=None, 
                     volume_id=None,
                     clim=None):
        self._init_ortho_vewports(coords=coords,
                                  volume_id=volume_id,
                                  clim=clim)
        self._set_slider_widgets()
        self._activate_widgets()
        self._activate_mouse()

    def _init_ortho_vewports(self, 
                             coords: tuple[float, float, float] = None,
                             volume_id: int = 0, 
                             clim: tuple[float, float] = None):
        params = self.parameters
        output = self.outputs['ortho']
        with output:
            try:
                output.clear_output(wait=True)
                self.plotter['main'] = OrthoPlotter(
                    self.selector['main'], 
                    coords=coords, 
                    coordinate_system=params.coordinate_system, 
                    volume_id=volume_id,
                    crosslines=params.crosslines,
                    crosslines_kwargs=params.crosslines_kwargs,
                    hide_spines=params.hide_spines,
                    clim=clim,
                    cmap=params.cmap,
                    contrast_percentile=params.contrast_percentile,
                    figsize=params.figsize,
                    dpi=params.dpi
                    )
                
                self.plotter['main'].plot()
                self.fig = self.plotter['main'].fig
                self.fig.canvas.toolbar_visible = False
                self.fig.canvas.header_visible = False
                self.fig.canvas.footer_visible = False
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Error in plotting: {e}")
        
    def _set_slider_widgets(self, image_item: str = 'main'):
        img = self.canonical_img
        # Set up the sliders for the three orthogonal planes
        extent = self.selector[image_item].get_world_extent()
        center = self.selector[image_item].get_world_center()
        for p in Plane:
            vmin, vmax = extent[p]
            gap = img.voxel_sizes[p.value]/2
            self.widgets.sliders['xyz'].append(FloatSlider(min=vmin+gap,
                                                           max=vmax-gap,
                                                           step=img.voxel_sizes[p.value], 
                                                           description=Plane.to_string(p).capitalize(),
                                                           value=center[p.value]))

        # Set up the volume slider if the image has 3 or more dimensions
        if img.ndim >= 3:
            self.widgets.sliders['vol'] = IntSlider(min=0, 
                                                    max=img.shape[-1]-1, 
                                                    step=1,
                                                    description='Volume', 
                                                    continuous_update=False)

    def _update_viewport(self, event):
        # Callback function to update the orthogonal views when sliders are changed
        x = self.widgets.sliders['xyz'][0].value
        y = self.widgets.sliders['xyz'][1].value
        z = self.widgets.sliders['xyz'][2].value
        vol = self.widgets.sliders['vol'].value if self.widgets.sliders['vol'] else 0
        for plot_item, plotter in self.plotter.items():
            if plot_item == 'main':
                plotter.update(coords=(x, y, z), volume_id=vol)
            else:
                plotter.update(coords=(x, y, z))

    def _activate_widgets(self):
        for slider in self.widgets.sliders['xyz']:
            slider.observe(self._update_viewport, names='value')
            slider.layout = Layout(flex='0 0.1 auto')

        if vol_slider := self.widgets.sliders.get('vol'):
            vol_slider.observe(self._update_viewport, names='value')
            vol_slider.layout = self.layouts['half_center']

    def _onclick(self, event):
        if event.inaxes is None:
            return
        sliders = self.widgets.sliders['xyz']
        selected = [event.inaxes == ax for ax in self.plotter['main'].fig.axes]
        selected_sliders = [s for i, s in enumerate(sliders) if not selected[i]]
        for i, val in enumerate([event.xdata, event.ydata]):
            if val is not None:
                slider = selected_sliders[i]
                rounded_val = slider.min + round((val - slider.min) / slider.step) * slider.step
                rounded_val = max(min(rounded_val, slider.max), slider.min)
                slider.value = rounded_val
                
    def _activate_mouse(self):
        self.cid['navigate'] = self.fig.canvas.mpl_connect('button_press_event', self._onclick)
    
    def show(self, 
             coords=None, 
             volume_id=None,
             clim=None):

        self.init_widgets(coords=coords, volume_id=volume_id, clim=clim)
        width = int(self.parameters.figsize[0] * self.parameters.dpi)
        
        viewport_layout = Layout(width=f'{width}px', justify_content='center', align_items='center')
        self.outputs['ortho'].layout = viewport_layout
        
        sliderbox = HBox(self.widgets.sliders['xyz'], layout=viewport_layout)
        if vol_slider := self.widgets.sliders['vol']:
            sliderbox = VBox([sliderbox, vol_slider], layout=viewport_layout)
        
        base_viewer = VBox([self.outputs['ortho'], sliderbox], layout=self.layouts['full_center'])
        # Display the GUI
        display(base_viewer)
