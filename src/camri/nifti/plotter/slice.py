from matplotlib.axes import Axes
import numpy as np


class SlicePlotter:
    """
    SlicePlotter is a utility class for visualizing 2D slices of NIfTI images using Matplotlib.

    This class provides methods to display image data, adjust visualization parameters such as
    contrast and colormap, and overlay crosslines for better spatial reference. It is designed
    to work with Matplotlib's Axes object and provides an interface for interactive updates
    to the displayed image.

    Attributes
    im : matplotlib.image.AxesImage
        The image object displayed on the Axes.
    _vmin : float
        The minimum value for the colormap (contrast lower limit).
    _vmax : float
        The maximum value for the colormap (contrast upper limit).
    vline : matplotlib.lines.Line2D
        The vertical crossline object (if added).
    hline : matplotlib.lines.Line2D
        The horizontal crossline object (if added).

    Methods
    -------
    set_cmap(cmap: str)
    set_contrast(vmin: float, vmax: float)
    update_data(data: np.ndarray)
        Update the image data displayed.
    update_extent(extent: tuple[float, float, float, float])
        Update the spatial extent of the image.
    add_crosslines(x, y, color: str = 'k', linestyle='--', linewidth=1, alpha: float = 0.5)
        Add crosslines at specified coordinates.
    set_xlim(xmin: float, xmax: float)
    set_ylim(ymin: float, ymax: float)
    update_crosslines(x, y)
        Update the position of the crosslines.
    draw()
        Redraw the figure canvas to reflect updates.
    """
    def __init__(self, 
                 ax: Axes, 
                 data: np.ndarray,
                 extent: tuple[float, float, float, float], 
                 clim: tuple[float, float] = None,
                 interpolation: str = 'nearest',
                 cmap: str = 'gray'):

        self.im = ax.imshow(data.T, origin='lower', extent=extent)
        ax.set_aspect('equal')
        self.set_contrast(clim[0], clim[1])
        self.set_cmap(cmap)
        self.set_interpolation(interpolation)

    def set_interpolation(self, interpolation: str):
        """
        Set the interpolation method for the image.

        Parameters
        ----------
        interpolation : str
            The interpolation method to use for the image.
        """
        self.im.set_interpolation(interpolation)

    def set_cmap(self, cmap: str):
        """
        Set the colormap for the image.

        Parameters
        ----------
        cmap : str
            The colormap to use for the image.
        """
        self.im.set_cmap(cmap)

    def set_contrast(self, vmin: float, vmax: float):
        """
        Set the contrast limits for the image.

        Parameters
        ----------
        vmin : float
            The minimum value for the colormap.
        vmax : float
            The maximum value for the colormap.
        """
        self.im.set_clim(vmin, vmax)

    def update_data(self, data: np.ndarray):
        """
        Update the image data.

        Parameters
        ----------
        data : np.ndarray
            The new data to display in the image.
        """
        self.im.set_data(data.T)

    def update_extent(self, extent: tuple[float, float, float, float]):
        """
        Update the extent of the image.

        Parameters
        ----------
        extent : tuple[float, float, float, float]
            The new extent for the image.
        """
        self.im.set_extent(extent)

    def get_crosslines_kwargs(self, color=None, linestyle=None, linewidth=None, alpha=None):
        return {
            'color': color or 'white',
            'linestyle': linestyle or '--',
            'linewidth': linewidth or 1,
            'alpha': alpha or 0.5
            }

    def add_crosslines(self, x, y, **crosslines_kwargs):
        """
        Adds crosslines (vertical and horizontal) to the plot at specified coordinates.

        Parameters:
            x (float): The x-coordinate for the vertical line.
            y (float): The y-coordinate for the horizontal line.
            color (str, optional): The color of the lines. Defaults to 'k' (black).
            linestyle (str, optional): The style of the lines (e.g., '--' for dashed). Defaults to '--'.
            linewidth (int, optional): The width of the lines. Defaults to 1.
            alpha (float, optional): The transparency level of the lines (0.0 is fully transparent, 
                                     1.0 is fully opaque). Defaults to 0.5.

        Returns:
            None
        """
        ax = self.im.axes
        kwargs = self.get_crosslines_kwargs(**crosslines_kwargs)
        self.vline = ax.axvline(x=x, **kwargs)
        self.hline = ax.axhline(y=y, **kwargs)

    def remove_crosslines(self):
        """
        Remove the crosslines from the plot.

        Returns:
            None
        """
        if hasattr(self, 'vline'):
            self.vline.remove()
            del self.vline
        if hasattr(self, 'hline'):
            self.hline.remove()
            del self.hline

    def set_xlim(self, xmin: float, xmax: float):
        """
        Set the x-axis limits for the image.

        Parameters
        ----------
        xmin : float
            The minimum x-axis limit.
        xmax : float
            The maximum x-axis limit.
        """
        self.im.axes.set_xlim(xmin, xmax)

    def set_ylim(self, ymin: float, ymax: float):
        """
        Set the y-axis limits for the image.

        Parameters
        ----------
        ymin : float
            The minimum y-axis limit.
        ymax : float
            The maximum y-axis limit.
        """
        self.im.axes.set_ylim(ymin, ymax)

    def update_crosslines(self, x, y, **crosslines_kwargs):
        """
        Update the cross lines to new coordinates.

        Parameters
        ----------
        x : float
            The new x-coordinate for the vertical line.
        y : float
            The new y-coordinate for the horizontal line.
        """
        if not hasattr(self, 'vline') or not hasattr(self, 'hline'):
            self.add_crosslines(x, y, **crosslines_kwargs)
        else:
            self.vline.set_xdata([x, x])
            self.hline.set_ydata([y, y])
            
            kwargs = self.get_crosslines_kwargs(**crosslines_kwargs)
            for line in (self.vline, self.hline):
                line.set_color(kwargs['color'])
                line.set_linestyle(kwargs['linestyle'])
                line.set_linewidth(kwargs['linewidth'])
                line.set_alpha(kwargs['alpha'])

    def draw(self):
        """
        Redraw the figure canvas.
        """
        self.im.figure.canvas.draw_idle()
