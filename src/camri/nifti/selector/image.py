from ..handler import Nifti1Handler
from ..plane import Plane
import numpy as np


class ImageSelector:  
    """
    A simple NIfTI image slicer class.

    The input image is assumed to be in canonical (RAS+) orientation for display.
    Orthogonal views (sagittal, coronal, axial) of a specified slice (or from a given volume index
    for 4D images) are displayed in one row. Each subplot shares an identical square viewport,
    whose side length is determined by the largest field-of-view among the image's three axes.

    The imshow "extent" is set based on the image's world coordinates so that any overlays will align
    correctly regardless of resolution.

    This class also caches the selected volume. The interactive view uses sliders in world coordinates
    (with an initial value of 0 if within the image extent) and overlays dashed cross lines.
    """
    def __init__(self, img: Nifti1Handler):
        """
        Initialize a NiftiSlicer instance.

        Takes a Nifti1Handler image, ensures it is in canonical orientation, and initializes
        the volume cache starting with volume 0. The image data and metadata are extracted
        and stored as instance attributes.

        Parameters
        ----------
        img : Nifti1Handler
            The input NIfTI image to be sliced. Will be converted to canonical orientation
            if not already in RAS+ format.
        """
        # Assume img is already in canonical orientation
        self.img = img.to_canonical()
        # Initialize with volume 0
        self._vol_id = -1
        self.volume_id = 0

    @property
    def volume_id(self):
        return self._vol_id

    @volume_id.setter
    def volume_id(self, new_id: int):
        """
        Set the current volume ID and update the cached data.

        When a new volume ID is set, extracts that volume's data and metadata including:
        - Raw image data array
        - Shape, affine matrix, and voxel sizes
        - Field of view and viewport dimensions

        Parameters
        ----------
        new_id : int
            Index of the volume to extract and cache
        """
        if new_id != self._vol_id:
            self._vol_id = new_id
            self._cache = self.img.extract_volume(new_id)
            self._data = self._cache.get_nibobj().get_fdata()
            self.shape = self._cache.shape
            self.affine = self._cache.affine
            self.voxel_sizes = self._cache.voxel_sizes
            self.field_of_view = np.array(self.shape[:3]) * self.voxel_sizes
            self.viewport_size = float(self.field_of_view.max())

    def _encode_plane(self, plane: str | int | Plane):
        """
        Convert a plane specification into a Plane enum.

        Parameters
        ----------
        plane : str or int
            Anatomical plane to extract ('axial', 'coronal', or 'sagittal')
        or a Plane enum instance
        """
        if isinstance(plane, str):
            plane = plane.lower()
            plane = Plane.from_string(plane)
        elif isinstance(plane, int):
            plane = Plane.from_value(plane)
        elif not isinstance(plane, Plane):
            raise ValueError("Plane must be 'axial', 'coronal', or 'sagittal' or a Plane enum.")
        return plane

    def _extract_slice(self, plane: str | int | Plane, slice_spec: float | int, coordinate_system: str):
        """
        Extract a 2D slice from the volume and calculate its display extent.

        Parameters
        ----------
        plane : str
            Anatomical plane to extract ('axial', 'coronal', or 'sagittal')
        slice_spec : float or int
            Location to slice at - either a world coordinate (mm) or voxel index
            depending on coordinate_system
        coordinate_system : str
            'world' for physical coordinates in mm, 'index' for voxel indices

        Returns
        -------
        slice_img : ndarray
            2D array containing the extracted slice data
        extent : list
            [xmin, xmax, ymin, ymax] coordinates in world space defining the slice boundaries

        Raises
        ------
        IndexError
            If the calculated slice index is outside the volume dimensions
        """
        # Determine the index value
        plane = self._encode_plane(plane)
            
        if coordinate_system == 'world':
            idx = int(round((slice_spec - self.affine[plane.value, 3]) / self.voxel_sizes[plane.value]))
        else:
            idx = int(slice_spec)
            
        if idx < 0 or idx >= self.shape[plane.value]:
            raise IndexError("Axial slice index out of range.")
        
        slice_img = self._data[Plane.get_indexer(plane, idx)]
        x_idx, y_idx = Plane.get_others(plane)
        x_min, y_min = self.affine[x_idx, 3], self.affine[y_idx, 3]
        x_max = x_min + self.voxel_sizes[x_idx]*self.shape[x_idx]
        y_max = y_min + self.voxel_sizes[y_idx]*self.shape[y_idx]
        extent = [x_min, x_max, y_min, y_max]
        return slice_img, extent

    def get_slice(self, plane: str | int | Plane, slice_spec: float | int, coordinate_system: str = 'index', volume_id: int = 0):
        """
        Get a 2D slice from the specified volume and anatomical plane.

        A high-level interface that handles volume caching and calls _extract_slice()
        to get the actual slice data and extent.

        Parameters
        ----------
        plane : str
            Anatomical plane to extract ('axial', 'coronal', or 'sagittal')
        slice_spec : float or int
            Location to slice at - either world coordinate or voxel index
        coordinate_system : str, optional
            'world' for physical coordinates, 'index' for voxel indices (default)
        vol_index : int, optional
            For 4D images, which volume to extract from (default 0)

        Returns
        -------
        slice_img : ndarray
            2D array containing the slice data
        extent : list
            [xmin, xmax, ymin, ymax] coordinates defining slice boundaries
        """
        if self.img.get_nibobj().ndim >= 4:
            self.volume_id = volume_id
        return self._extract_slice(plane, slice_spec, coordinate_system)
    
    def get_world_extent(self):
        """
        Return the world extent of the image, computed as the product
        of the voxel sizes and the image dimensions (first three axes).

        Returns
        -------
        dict
            Each plane's min/max coordinates:
            {
                'sagittal': (min, max),
                'coronal': (min, max), 
                'axial': (min, max)
            }
        """
        extents = {}
        for p in Plane:
            min_val = self.affine[p.value, 3]
            max_val = min_val + self.voxel_sizes[p.value] * self.shape[p.value]
            extents[p] = (min_val, max_val)
        return extents
    
    def get_world_center(self):
        """
        Return the world center of the image, computed as the average of the world extent.
        """
        extents = self.get_world_extent()
        return tuple(np.mean(list(extents.values()), axis=1))