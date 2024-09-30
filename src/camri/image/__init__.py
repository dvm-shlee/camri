from .deprecated import OrthoViewer, singlerow_orthoplot
from .handler import Handler
from .utils import compose_nifti, merge_affine
from ..prep.rsfc import corr_with, get_cluster_coordinates
from nibabel.nifti1 import Nifti1Image
from .viewer import SliceView
import numpy as np
import matplotlib.pyplot as plt


class Image(Handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def calc_connectivity_at_seed(self, seed_coord, size, nn_level=3, 
                                  coord_type='xyz', mask_img=None, 
                                  gen_seed_img=False):
        """ Calculate correlation with seed map
        
        Args:
            seed_coord(tuple[float, float, float]): coordinate of seed
            size(int): if coord_type == 'xyz' size is mm, if 'itk' then size is scale to 1 voxel
            nn_level(int): nearest neighbor factor (same as ReHo)
            coord_type(str): either 'xyz' or 'itk', (default = 'xyz')
        Returns:
            seed(Handler): seed image
            seedmap(Handler): connectivity results
        """
        if coord_type not in ['xyz', 'itk']:
            raise TypeError('The type of coordinate must be either "xyz" or "itk"')
        if coord_type == 'xyz':
            seed_coord = self.xyz_to_itk(*seed_coord)
            size = int(size / self.resol.min())
        data = self.get_data()
        if mask_img:
            if isinstance(mask_img, Handler):
                mask_idx = np.nonzero(mask_img.get_data())
            elif isinstance(mask_img, Nifti1Image):
                mask_idx = np.nonzero(mask_img.dataobj)
            else:
                mask_idx = np.nonzero(data.mean(-1))
        else:
            mask_idx = np.nonzero(data.mean(-1))
            
        seed_mask = tuple(np.transpose(get_cluster_coordinates(seed_coord, size=size, nn_level=nn_level)))
        data = self.get_data()[mask_idx]
        try:
            seed = self.get_data()[seed_mask].mean(0)
        except Exception as e:
            raise Exception(e)
        r = corr_with(data, seed[np.newaxis, :])
        nifti = compose_nifti(r, self._nifti, mask_idx=mask_idx)
        if gen_seed_img:
            seed_data = np.zeros(data.shape[:3])
            seed_data[seed_mask] = 1
            return Image(nifti), Image(compose_nifti(seed_data, self._nifti, masi_idx=mask_idx))
        return Image(nifti)

    def transform_to_ras(self):
        return Image(self.ras_nifti)
    
    def crop(self, l=0, r=0, p=0, a=0, i=0, s=0):
        slices = []
        shift = []
        for j, c in enumerate(self.orientation_codes[:3]):
            if c == "L":
                l = self.shape[j] - l
                slices.append(slice(r, l))
                shift.append(r)
            elif c == "R":
                r = self.shape[j] - r
                slices.append(slice(l, r))
                shift.append(l)
            elif c == "P":
                p = self.shape[j] - p
                slices.append(slice(a, p))
                shift.append(a)
            elif c == "A":
                a = self.shape[j] - a
                slices.append(slice(p, a))
                shift.append(p)
            elif c == "I":
                i = self.shape[j] - i
                slices.append(slice(s, i))
                shift.append(s)
            else:
                s = self.shape[j] - s
                slices.append(slice(i, s))
                shift.append(i)
        
        origin = self.origin.copy()
        origin += self.orthogonal.dot(np.array(shift) * self.resol)
        dataobj_cropped = self.get_data()[tuple(slices)]
        affine = merge_affine(self.rotate, origin)
        nifti = compose_nifti(dataobj_cropped, self._nifti, affine)
        return Image(nifti)
    
    def show(self, mode='slice', return_fig=False, **kwargs):
        if mode == 'slice':
            max_columns = kwargs.get('max_columns') or 5
            frame_id = kwargs.get('frame_id') or 0
            axis = kwargs.get('axis')
            indexes = kwargs.get('indexes')
            indexes = indexes if isinstance(indexes, (tuple, list)) else [indexes]
            axes_kw = kwargs.get('axes_kw')
            imshow_kw = kwargs.get('imshow_kw')
            subplots_kw = kwargs.get('subplots_kw') or {}
            
            if axis is None:
                plane = kwargs.get('plane') or 'coronal'
                coords = kwargs.get('coords') or 0.0
                coords = coords if isinstance(coords, (list, tuple)) else [coords]
                num_slices = len(coords)
            else:
                num_slices = len(indexes)
                    
            num_row = int(num_slices / max_columns) + (num_slices % max_columns)
            num_column = num_slices if num_slices < max_columns else max_columns
            
            fig, axes = plt.subplots(num_row, num_column, **subplots_kw)
            axes = axes.flatten() if num_slices > 1 else [axes]
            for i in range(num_slices):
                if len(coords) < num_slices:
                    index = indexes[i]
                    coord = None
                else:
                    index = None
                    coord = coords[i]
                SliceView(self, frame_id, 
                          axis=axis, index=index,
                          plane=plane, coord=coord).show(ax=axes[i], 
                                                         axes_kw=axes_kw, 
                                                         imshow_kw=imshow_kw)
                axis('off')
            for i in range(num_slices, len(axes)):
                axes[i].axis('off')
            plt.tight_layout()
            if return_fig:
                return fig
            
        else:
            raise NotImplementedError('Only slice view is available')


def load(path):
    return Image(path)


__all__ = ['Image', 'load']