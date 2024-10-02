import numpy as np
from .utils import validate_nifti_input, compose_nifti
from .utils import split_affine, merge_affine
from .utils import remove_element_at
from .utils import calculate_transform_matrix


class Handler:
    """ Image Handler object to help image handling easier for image analsys and reviewing the image
    """
    def __init__(self, nifti, **kwargs):
        self._nifti = validate_nifti_input(nifti)
        
        
        self.reset_orient()
        self._affine_params = kwargs.get('affine_params', {'decimals': 3, 
                                                           'deoblique': True})
        if self._affine_params['deoblique']:
            self.deoblique()

    @property
    def dim(self):
        return self._nifti.header['dim'][0]
    
    @property
    def shape(self):
        return self._nifti.header['dim'][1:4]
    
    @property
    def resol(self):
        return self._nifti.header['pixdim'][1:4]
    
    @property
    def num_frames(self):
        if self.dim == 3:
            return 1
        elif self.dim == 4:
            # functional images or vector images (DTI)
            return self._nifti.header['dim'][4]
        elif self.dim == 5 and self._nufti.header[3] == 1:
            # afni's statistic bricks
            return self._nifti.header['dim'][5]
        else:
            # other exception cases, this case, the handler may not properly works
            return self._nufti.header['dim'][4:]
    
    @property
    def rotate(self):
        return self._rotate
    
    @property
    def origin(self):
        return self._origin
    
    @property
    def affine(self):
        return merge_affine(self.rotate, self.origin)
    
    @rotate.setter
    def rotate(self, rotate):
        # Ensure that rotate is a 3x3 array (or list of lists)
        if isinstance(rotate, (list, tuple, np.ndarray)) and np.shape(rotate) == (3, 3):
            self._rotate = np.array(rotate)  # Convert to numpy array for consistency
        else:
            raise ValueError("rotate must be a 3x3 array, list, or tuple.")
    
    @origin.setter
    def origin(self, origin):
        if isinstance(origin, (list, tuple, np.ndarray)) and len(origin) == 3:
            self._origin = np.array(origin)  # Convert to numpy array for consistency
        else:
            raise ValueError("origin must be a list, tuple, or array with exactly 3 elements.")
    
    @property
    def orthogonal(self):
        """return orthogonal matrix represent the orthogonal transformation required to reorient matrix to RAS+
        The inverse of this matrix can trasnform RAS+ orientation into original orientation, descaled, deobliqued
        """
        scale_factor = np.linalg.inv(np.diag(self.resol))
        return np.round(self.rotate.dot(scale_factor), decimals=0)
    
    @property
    def orient_info(self):
        """return the correspond eucledian coordinate axis and subject position toward increase of 
        matrix index based on RAS+ oriented coordinate space.
        """
        itk_code = []
        for i, direct in enumerate(self.axis_direction):
            code = ['i', 't', 'k'][i]
            if direct < 0:
                itk_code.append(f'-{code}')
            else:
                itk_code.append(code)
        return dict(zip(itk_code, [l for l in zip(self.xyz_axis_order, self.orientation_codes)]))
    
    @property
    def axis_direction(self):
        """Returns the index increment direction based on the current axis orientation order, 
        assuming the increment follows RAS (Right-Anterior-Superior)."""
        return self.orthogonal.sum(0)
    
    @property
    def ras_affine(self):
        ras_resolution = np.abs(self.orthogonal.dot(self.resol))
        ras_rotate = np.diag(ras_resolution)
        ras_origin = -np.abs(self.origin)
        return merge_affine(ras_rotate, ras_origin)
    
    @property
    def transform_matrix_to_ras(self):
        return calculate_transform_matrix(self.ras_affine, self.affine)
    
    @property
    def itk_to_ras_axis_order(self):
        """ Returns the index for axis reordering to convert the axis order from the current image's axis order (itk)
        to the standard RAS (Right-Anterior-Superior) in xyz coordinate, disregarding directionality.
        """
        return np.nonzero(self.orthogonal)[1]
    
    @property
    def ras_to_itk_axis_order(self):
        """ Returns the index for axis reordering to convert from xyz coordinate in RAS (Right-Anterior-Superior) 
        axis to the array's itk matrix axis order, disregarding directionality.
        """
        return np.nonzero(np.linalg.inv(self.orthogonal))[1]
    
    @property
    def ras_axis_direction(self):
        """Returns the index increment direction in RAS+ order (Right-Anterior-Superior positive)."""
        return self.orthogonal.sum(1)
    
    @property
    def orientation_codes(self):
        """Capital letter indicates direction toward value increment.
        Example:
            RAS meaning data index increment toward Left->Right, Posterior->Anterior, Inferior->Superior
        """
        return tuple([['LR','PA','IS'][i][(self.ras_axis_direction > 0).astype(int)[i]] for i in self.ras_to_itk_axis_order])
    
    @property
    def xyz_axis_order(self):
        return tuple(np.array(['x', 'y', 'z'])[self.ras_to_itk_axis_order].tolist())
    
    @property
    def indexed_matrix_plane_order(self):
        slice_axis_to_plane = {'x':'sagital', 'y':'coronal', 'z':'axial'}
        return tuple([slice_axis_to_plane[axis_name] for axis_name in self.xyz_axis_order])
    
    @property
    def meshgrid(self):
        """Returns the meshgrid of the image in the (x, y, z) coordinate space based on RAS+ orientation.
        """
        origin = self.origin[self.ras_to_itk_axis_order]
        stop = origin + self.axis_direction * self.shape * self.resol
        meshgrid = {}
        for idx, axis_code in enumerate(self.xyz_axis_order):
            meshgrid[axis_code] = np.linspace(origin[idx],
                                              stop[idx],
                                              self.shape[idx],
                                              endpoint=False)
        return meshgrid
    
    @property
    def ras_data(self):
        target_order = self.itk_to_ras_axis_order
        if self.dim > 3:
            target_order = np.append(target_order, np.arange(3, self.dim+1))
        ras_dataobj = np.transpose(self.get_data(), target_order)
        slicer = tuple(slice(None, None, int(step)) for step in self.ras_axis_direction)
        return ras_dataobj[slicer]
        
    @property
    def ras_nifti(self):
        return compose_nifti(self.ras_data, self._nifti, self.ras_affine)
        
    def reset_orient(self):
        self._rotate, self._origin = split_affine(self._nifti.affine)
        
    def is_oblique(self):
        return np.nonzero(self.rotate)[0].shape[0] != 3
    
    def deoblique(self, affine_only=True):
        if self.is_oblique():
            print("The image is oblique. ", end="")
            rotate = self.orthogonal.dot(np.diag(self.resol))
            if affine_only:
                print("Deobliquing the affine information only...")
                self.rotate = rotate
                print("Affine matrix deobliqued.")
            else:
                raise NotImplementedError("Deobliquing image matrix is not supported yet...")
                # print("Applying deoblique transformation...")
                # TODO: apply transform to deoblique matrix 
                # print("Deoblique transformation applied.")
    
    def xyz_to_itk(self, x, y, z):
        """Convert RAS+ (x, y, z) ordered coordinates to matrix coordinates (i, t, k)."""
        itk = (np.array([x, y, z, 0]) - self.affine[:, 3]).dot(np.linalg.inv(self.affine))[:3]
        return tuple(itk.astype(int))
    
    
    def itk_to_xyz(self, i, t, k):
        """Convert matrix coordinates (i, t, k) to RAS+ (x, y, z) ordered coordinates."""
        xyz = np.array([i, t, k, 0]).dot(self.affine)[:3] + self.affine[:3, 3]
        return tuple(xyz)
    
    def get_data(self):
        """ Return whole data matrix
        """
        return self._nifti.get_fdata()
    
    def get_volume(self, frame_index=0):
        """ Retern specific time data.
        """
        dataobj = self._nifti.dataobj
        if self.dim > 3:
            max_index = self.shape[-1]-1
            if frame_index > max_index:
                raise IndexError(f"index {frame_index} is out of bounds for axis {self.dim-1} with size {max_index}")
            return np.asarray(dataobj[..., frame_index]).squeeze()
        else:
            return np.asarray(dataobj)
    
    def get_plane_by_itk_index(self, slice_index, axis=2, frame_index=0):
        """ Retern 2D plane image at given axis id and index.
        """
        volume = self.get_volume(frame_index)
        slicer = [slice(None, None, None)] * 3
        slicer[axis] = slice_index
        
        orient_codes = remove_element_at(self.orientation_codes, axis)
        xyz_order = remove_element_at(self.xyz_axis_order, axis)
        
        extent = []
        for axis_id, axis_name in enumerate(xyz_order):
            pad = self.resol[axis_id]/2 * self.axis_direction[axis_id]
            meshgrid = self.meshgrid[axis_name]
            extent.extend([meshgrid[0] - pad, meshgrid[-1] + pad])
        return dict(zip(xyz_order, orient_codes)), extent, volume[tuple(slicer)]
        
    def get_plane_by_xyz_coord(self, coord, plane='axial', frame_index=0):
        """ Retern 2D plane image at given coordinate and plane name.
        """
        plane_to_axis = {'axial': 'z', 'coronal': 'y', 'sagital':'x'}
        axis = plane_to_axis[plane]
        axis_id = list(self.xyz_axis_order).index(axis)
        itk = self.xyz_to_itk(*(np.ones(3) * coord).tolist())
        slice_index = itk[axis_id]
        return self.get_plane_by_itk_index(slice_index, axis_id, frame_index)
    
    def __contains__(self):
        """ In operation
        return data in mask
        
        
        """
        pass