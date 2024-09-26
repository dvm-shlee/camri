import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from scipy.ndimage import zoom
from ._imgtool import *
from .handler import Handler
from nibabel.nifti1 import Nifti1Image

def singlerow_orthoplot(nibobj, slices=None, frame_id=0, grid=0, 
                        annotation=True, annotation_color=None, annotation_size=None, 
                        crossline=True, crossline_color=None, crossline_width=None,
                        **kwargs):
    # default parameters
    kwargs.setdefault('origin', 'lower')
    kwargs.setdefault('aspect', 1)
    kwargs.setdefault('cmap', 'gray')
    kwargs.setdefault('dpi', 200)
    kwargs.setdefault('interpolation', 'nearest')
    kwargs.setdefault('figsize', None)
    
    annotation_size = annotation_size or 6
    
    if isinstance(nibobj, Nifti1Image):
        nibobj = nibobj
    elif isinstance(nibobj, Handler):
        nibobj = nibobj._nifti
    else:
        raise TypeError(f"Image file must be Nifti1Image object or Handler object.")

    # load image
    ras_nii = reorient_to_ras(nibobj)
    if nibobj.header['dim'][0] > 3:
        dataobj = ras_nii.dataobj[..., frame_id]
    else:
        dataobj = ras_nii.dataobj
    
    # image
    ratio = get_resol(ras_nii) 
    ratio /= ratio.min() # ratio for resolution
    img = zoom(np.asarray(dataobj).squeeze(), ratio, order=0)
    slices = np.array(slices) * ratio if slices else (np.array(img.shape[:3]) / 2)
    slices = slices.astype(int)
    
    # colors
    cmap = plt.get_cmap(kwargs['cmap'])
    kwargs['vmin'] = kwargs.get('vmin', img.min())
    kwargs['vmax'] = kwargs.get('vmax', img.max())
    norm = colors.Normalize(vmin=kwargs['vmin'], vmax=kwargs['vmax'])
    zero_color = cmap(norm(0))
    comp_color = tuple([1-c for c in zero_color[:3]])
    
    kwargs.setdefault('facecolor', zero_color)
    face_color = colors.to_rgb(kwargs.pop('facecolor'))
    font_color = tuple([1-c for c in face_color[:3]])
    
    # plotting
    sliced_imgs = {
        0: ('sagital (y, z)', img[slices[0],:,:].T),
        1: ('coronal (x, z)', img[:,slices[1],:].T),
        2: ('axial (x, y)', img[:,:,slices[2]].T)
    }
    output_shape = [max(img.shape)] * 2
    fig, ax = plt.subplots(1, 3, 
                           facecolor=face_color,
                           figsize = kwargs.pop('figsize'),
                           dpi = kwargs.pop('dpi'))
    for i, (label, sliced_img) in sorted(sliced_imgs.items()):
        shift = np.subtract(sliced_img.shape, output_shape)/2
        sliced_ratio = [e for j, e in enumerate(ratio) if j != i]
        ax[i].imshow(sliced_img, **kwargs)
        ax[i].set_title(label, color=font_color, fontsize=annotation_size + 1, fontweight='medium')
        ax[i].set_facecolor(face_color)
        
        for spine in ax[i].spines.values():
            spine.set_visible(False)
        if grid:
            ax[i].xaxis.set_ticks_position('bottom')  # Only show x-ticks on the bottom
            ax[i].yaxis.set_ticks_position('left')    # Only show y-ticks on the left
            ax[i].xaxis.set_major_locator(ticker.MultipleLocator(grid*sliced_ratio[0]))
            ax[i].yaxis.set_major_locator(ticker.MultipleLocator(grid*sliced_ratio[1]))
            ax[i].tick_params(axis='both', which='major', direction='out', colors=font_color,
                              labelleft=False, labelbottom=False, length=2, width=0.5)
        else:
            ax[i].axis('off')
        xlim = [shift[1], sliced_img.shape[1]-shift[1]]
        ylim = [shift[0], sliced_img.shape[0]-shift[0]]
        ax[i].set_xlim(xlim)
        ax[i].set_ylim(ylim)
        
        # Annotation for orientation 
        ori_labels = {
            0: ("P", "A", "I", "S"),
            1: ("L", "R", "I", "S"),
            2: ("L", "R", "P", "A")
        }
        if annotation:
            left_label, right_label, bottom_label, top_label = ori_labels[i]
            # xaxis
            width = sliced_img.shape[1] / (xlim[1] - xlim[0])
            if width > 0.8:
                annotation_color = comp_color
            else:
                annotation_color = font_color
            ax[i].text(0.05, 0.5, left_label, fontsize=annotation_size, 
                       ha='center', va='center', color=annotation_color,
                       rotation=0, transform=ax[i].transAxes)
            ax[i].text(0.95, 0.5, right_label, fontsize=annotation_size, 
                       ha='center', va='center', color=annotation_color, 
                       rotation=0, transform=ax[i].transAxes)
            # yaxis
            height = sliced_img.shape[0] / (ylim[1] - ylim[0])
            if height > 0.8:
                annotation_color = comp_color
            else:
                annotation_color = font_color
            ax[i].text(0.5, 0.05, bottom_label, fontsize=annotation_size, 
                       ha='center', va='center', color=annotation_color,
                       rotation=0, transform=ax[i].transAxes)
            ax[i].text(0.5, 0.95, top_label, fontsize=annotation_size, 
                       ha='center', va='center', color=annotation_color,
                       rotation=0, transform=ax[i].transAxes)
        if crossline:
            sliced_loc = [e for j, e in enumerate(slices) if j != i]
            crossline_color = crossline_color or comp_color
            crossline_width = crossline_width or 0.5
            ax[i].axvline(x=sliced_loc[0], color=crossline_color, linestyle='--', linewidth=crossline_width)
            ax[i].axhline(y=sliced_loc[1], color=crossline_color, linestyle='--', linewidth=crossline_width)
    return fig


class OrthoViewer:
    view_order = ['sagital', 'coronal', 'axial']
    view_plain = ['(y,z)', '(x,z)', '(x,y)']
    
    def __init__(self, main_nii, underlay_nii=None, **kwargs):
        self.main_nii = main_nii
        self.main_img, self.main_meshgrid, self.main_resol = gridding(main_nii)
        self.main_thr = None
        self.pos_only = np.nonzero(self.main_img < 0)[0].shape == 0
        self.vmax = self.main_img.max()
        self.vmin = 0 if self.pos_only else -self.vmax
        
        if isinstance(underlay_nii, nib.Nifti1Image):
            self.set_underlay(underlay_nii)
        else:
            self.has_underlay = False
        self.set_facecolor()
        self.set_annotation_params()
        self.set_transparent_background()
        self.set_scalebar_size()
        
    def get_compcolor(self, color):
        return tuple([1-c for c in colors.to_rgb(color)])

    def _get_fov(self, decimals=3):
        edge = self.main_resol / 2
        grids = self.main_meshgrid.values()
        fov =  np.round([(a[0]-edge[0], a[-1]+edge[1]) for i, a in enumerate(grids)], 
                        decimals=decimals)
        return fov
    
    def get_fov(self, decimals=3):
        fov = self._get_fov(decimals)
        return {'sagital': (fov[1], fov[2]),
                'coronal': (fov[0], fov[2]),
                'axial': (fov[0], fov[1])}
    
    def get_cubic_fov(self):
        fov = self._get_fov()
        fov_size = np.abs(fov[:, 1] - fov[:, 0])
        size_diff = fov_size.max() - fov_size
        cubic_fov = fov.copy()
        cubic_fov[:, 0] -= size_diff/2
        cubic_fov[:, 1] += size_diff/2
        return {'sagital': (cubic_fov[1], cubic_fov[2]),
                'coronal': (cubic_fov[0], cubic_fov[2]),
                'axial': (cubic_fov[0], cubic_fov[1])}
    
    def get_sliced_img(self, itk_indice, frame_id):
        img = self.main_img.copy()
        if len(img.shape) > 3:
            img = img[..., frame_id]
        if self.transparent_background in ['both', 'main']:
            img = np.ma.masked_values(img, 0)
        if self.main_thr:
            img = np.ma.masked_where((img > -self.main_thr) & (img < self.main_thr), img)
        sag = img[itk_indice[0], :, :].T
        cor = img[:, itk_indice[1], :].T
        axi = img[:, :, itk_indice[2]].T
        return {'sagital': sag,
                'axial': axi,
                'coronal': cor}
    
    def get_indice_from_coord(self, xyz_coord):
        itk_indice = []
        for i, axis in enumerate(['x', 'y', 'z']):
            itk_indice.append(np.argmin(np.abs(self.main_meshgrid[axis] - xyz_coord[i])))
        return itk_indice
    
    def get_coord_from_indice(self, itk_indice):
        xyz_coord = []
        for i, axis in enumerate(['x', 'y', 'z']):
            coord = self.main_meshgrid[axis][itk_indice[i]] + self.main_resol[i]/2
            xyz_coord.append(np.round(coord, decimals=3))
        return xyz_coord
    
    def _get_sliced_underlay(self, itk_indice):
        coord_info = get_target_coord(self.main_meshgrid, self.ulay_meshgrid, itk_indice)
        slicers = get_ortho_slicer(*coord_info)
        if self.transparent_background in ['both', 'underlay']:
            ulay_img = np.ma.masked_values(self.ulay_img.copy(), 0)
        else:
            ulay_img = self.ulay_img.copy()
        sag = ulay_img[slicers['sagital']].T
        cor = ulay_img[slicers['coronal']].T
        axi = ulay_img[slicers['axial']].T
        return {'sagital': sag,
                'coronal': cor,
                'axial': axi}
    
    def set_scalebar_size(self, num_voxels=5):
        self.scalebar_size = num_voxels
    
    def set_facecolor(self, color='white'):
        self.facecolor = colors.to_rgb(color)
        self.set_annotation_params()
    
    def set_underlay(self, underlay_nii):
        self.ulay_nii = underlay_nii
        self.ulay_img, self.ulay_meshgrid, self.ulay_resol = gridding(underlay_nii)
        img = self.ulay_img.copy()
        masked = img[np.nonzero(img)]
        self.ulay_vminmax = (np.percentile(masked, 0.2), np.percentile(masked, 99.8))
        self.has_underlay = True
        
    def set_annotation_params(self, size=6, color=None):
        self.annot_size = size
        self.annot_color = color or self.get_compcolor(self.facecolor)
    
    def set_transparent_background(self, which='both'):
        """ which can be: main, underlay, both, off
        """
        self.transparent_background = which
    
    def set_threshold(self, thr):
        self.main_thr = thr
    
    def _set_plots(self, colorbar, **kwargs):
        width_ratios = [1, 1, 1, 0.03, 0.07] if colorbar else [1, 1, 1]
        kwargs.setdefault('facecolor', self.facecolor)
        kwargs.setdefault('figsize', np.array([6, 2.4]) if colorbar else np.array([6, 3]))
        kwargs.setdefault('dpi', 150)
        kwargs.setdefault('gridspec_kw', {
            'width_ratios': width_ratios, 'height_ratios': [1],
            'wspace': 0, 'hspace': 0
        })
        kwargs.setdefault('constrained_layout', False)
        self.figure, self.axes = plt.subplots(1, len(width_ratios), **kwargs)
        slice_stop = 4 if len(width_ratios) > 3 else 3
        for ax in self.axes[:slice_stop]:
            ax.axis('off')
    
    def set_title(self, title, **kwargs):
        self.figure.suptitle(title, **kwargs)
    
    def set_cbar(self, cmap, vmin, vmax, n_colors=256):
        ax = self.axes[4]
        cmap = cmap if isinstance(cmap, mpl.colors.Colormap) else plt.get_cmap(cmap)
        colors = np.expand_dims(cmap(np.linspace(0, 1, n_colors)), axis=1)
        pos = ax.get_position()
        new_pos = [pos.x0, pos.y0 + 0.2, pos.width, pos.height - 0.4]
        ax.set_position(new_pos)
        ax.imshow(colors, origin='lower', aspect='auto', extent=[0, 1, vmin, vmax])
        ax.tick_params(left=False, bottom=False, right=True, 
                       labelleft=False, labelbottom=False, labelright=True, 
                       labelsize=self.annot_size, labelcolor=self.annot_color, 
                       width=0.5, length=1, pad=2)
        ax.set_yticks([vmin, 0, vmax])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    
    def set_crossline(self, xyz_coord, **kwargs):
        fov = self.get_fov()
        for i, ax in enumerate(self.axes[:3]):
            view_plain = self.view_order[i]
            xlim, ylim = fov[view_plain]
            if view_plain == 'sagital':
                x, y = xyz_coord[1], xyz_coord[2]
            elif view_plain == 'axial':
                x, y = xyz_coord[0], xyz_coord[1]
            else:
                x, y = xyz_coord[0], xyz_coord[2]
            ax.plot([x, x], ylim, color=self.annot_color, linestyle='--', linewidth=0.5, alpha=0.7)
            ax.plot(xlim, [y, y], color=self.annot_color, linestyle='--', linewidth=0.5, alpha=0.7)
            
    def set_scalebar(self):
        xlim = self.get_cubic_fov()['axial'][0]
        xsize = xlim[1] - xlim[0]
        scale_size = np.round(xsize * 0.1, decimals=0)
        scale_ratio = scale_size / xsize

        min_resol = self.main_resol.min()
        xlim, ylim = self.get_fov()['axial']
        ax = self.axes[2]
        ax.plot([0.85, 0.85 + scale_ratio], [0.05]*2, 
                          color=self.annot_color, linewidth=1, transform=ax.transAxes)
        ax.text(0.85 + scale_ratio/2, 0.12, f'{scale_size}mm', ha='center', va='top', 
                size=self.annot_size, color=self.annot_color, transform=ax.transAxes)
        
    def set_annotation(self, itk_indice):
        ori_labels = {
            'sagital': ("P", "A", "I", "S"),
            'coronal': ("L", "R", "I", "S"),
            'axial': ("L", "R", "P", "A")
        }
        axis_name = ['x', 'y', 'z']
        xyz_coord = self.get_coord_from_indice(itk_indice)
        cubic_fov = self.get_cubic_fov()
        for i, ax in enumerate(self.axes[:3]):
            view_plain = self.view_order[i]
            left_label, right_label, bottom_label, top_label = ori_labels[view_plain]
            ax.text(0.05, 0.5, left_label, fontsize=self.annot_size, 
                       ha='center', va='center', color=self.annot_color,
                       rotation=0, transform=ax.transAxes)
            ax.text(0.95, 0.5, right_label, fontsize=self.annot_size, 
                       ha='center', va='center', color=self.annot_color, 
                       rotation=0, transform=ax.transAxes)
            # yaxis
            ax.text(0.5, 0.05, bottom_label, fontsize=self.annot_size, 
                       ha='center', va='center', color=self.annot_color,
                       rotation=0, transform=ax.transAxes)
            ax.text(0.5, 0.95, top_label, fontsize=self.annot_size, 
                       ha='center', va='center', color=self.annot_color,
                       rotation=0, transform=ax.transAxes)
            ax.text(0.5, -0.1, f"{axis_name[i]}={xyz_coord[i]}mm ({itk_indice[i]}/{self.main_img.shape[i]})", 
                    size=self.annot_size, color=self.annot_color, 
                    rotation=0, transform=ax.transAxes,
                    ha='center', va='center')
    
    def show(self, xyz_coord=None, itk_indice=None, frame_id=0, 
             annotation=True, colorbar=True, crossline=True, scalebar=True,
             figure_kw=None, underlay_kw=None, **kwargs):
        if xyz_coord:
            if itk_indice:
                raise ValueError('"xyz_coord" and "itk_indice" cannot be used together. Please provide only one.')
            itk_indice = self.get_indice_from_coord(xyz_coord)
        else:
            if not itk_indice:
                xyz_coord = [0, 0, 0]
                itk_indice = self.get_indice_from_coord(xyz_coord)
            else:
                xyz_coord = self.get_coord_from_indice(itk_indice)
        
        # figure prep
        figure_kw = figure_kw or {}
        self._set_plots(colorbar=colorbar, **figure_kw)
        
        # viewport configuration
        kwargs.setdefault('cmap', 'jet')
        kwargs.setdefault('vmin', self.vmin)
        kwargs.setdefault('vmax', self.vmax)
        fov = self.get_fov()
        cubic_fov = self.get_cubic_fov()
        sliced_img = self.get_sliced_img(itk_indice, frame_id)
        # load underlay image
        if self.has_underlay:
            underlay_img = self._get_sliced_underlay(itk_indice)
        else:
            underlay_img = None
            
        # plot image
        self.subplots = []
        for axes_id, view_plain in enumerate(self.view_order):
            img = sliced_img[view_plain]
            ax = self.axes[axes_id]
            ax.set_title(f"{view_plain} {self.view_plain[axes_id]}", 
                         size=self.annot_size, color=self.annot_color)
            
            xrange, yrange = fov[view_plain]
            extent=[xrange[0], xrange[1], yrange[0], yrange[1]]
            # plot underlay image
            if underlay_img is not None:
                underlay_kw = underlay_kw or {}
                underlay_kw.setdefault('cmap', 'gray')
                underlay_kw.setdefault('vmin', self.ulay_vminmax[0])
                underlay_kw.setdefault('vmax', self.ulay_vminmax[1])
                ax.imshow(underlay_img[view_plain], origin='lower', extent=extent, **underlay_kw)
            self.subplots.append(ax.imshow(img, origin='lower', 
                                           extent=extent, 
                                           **kwargs))
            ax.set_xlim(cubic_fov[view_plain][0])
            ax.set_ylim(cubic_fov[view_plain][1])
        if colorbar:
            self.set_cbar(kwargs['cmap'], kwargs['vmin'], kwargs['vmax'])
        if crossline:
            self.set_crossline(xyz_coord)
        if scalebar:
            self.set_scalebar()
        if annotation:
            self.set_annotation(itk_indice)
