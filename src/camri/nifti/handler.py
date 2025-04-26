from camri.utils.nifti import *
import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates


class Nifti1Handler:
    """
    Extended functionality for nibabel NIfTI images.
    
    This wrapper class provides additional methods for processing NIfTI images,
    such as extracting a specific volume from a 4D image, converting to canonical (RAS+)
    orientation, cropping, resampling while preserving world coordinates, and converting
    voxel indices between non-canonical and canonical spaces.
    """
    def __init__(self, img):
        """
        Initialize the Nifti1Handler object with a nibabel NIfTI image.
        
        Parameters:
            img (nibabel.Nifti1Image): The input NIfTI image.
        """
        if not isinstance(img, nib.Nifti1Image):
            raise TypeError("img must be a nibabel.Nifti1Image")
        self.img = img

    @property
    def ndim(self):
        """
        Return the number of dimensions of the image data.
        """
        return self.img.ndim

    @property
    def affine(self) -> np.ndarray:
        """
        Return the affine transformation matrix of the image data.
        Small values below floating point precision are set to zero.

        Returns
        -------
        np.ndarray
            4x4 affine matrix defining the image-to-world coordinate transform
        """
        affine = np.asarray(self.img.affine, dtype=np.float64)
        eps = np.finfo(float).eps
        return np.where(np.abs(affine) < eps, 0, affine)
        
    @property
    def shape(self):
        """
        Return the shape of the image data.
        """
        return self.img.shape

    @property
    def voxel_sizes(self):
        """
        Return the voxel sizes computed from the affine matrix.
        Uses np.finfo(float).eps to handle floating point precision.
        """
        raw_sizes = np.linalg.norm(self.img.affine[:3, :3], axis=0)
        eps = np.finfo(float).eps
        return np.where(np.abs(raw_sizes) < eps, 0, raw_sizes)

    @property
    def field_of_view(self):
        """
        Return the field of view (FOV) of the image, computed as the product
        of the voxel sizes and the image dimensions (first three axes).
        
        'field_of_view' represents the physical extent of the image.
        """
        return tuple(np.array(self.shape[:3]) * self.voxel_sizes)

    @property
    def header(self):
        """
        Return the header of the image.
        
        The header contains metadata about the image, such as data type, dimensions,
        and affine transformation.
        """
        return self.img._header

    @property
    def dataobj(self):
        """
        Return the data object of the image.
        
        The data object is a memory-mapped array that allows efficient access to the image data.
        """
        return self.img._dataobj

    def get_empty(self):
        """
        Return a reference image for the current NIfTI image.
        
        The reference image is a 3D image with the same affine and header as the original image.
        """
        header = self.img.header.copy()
        header.set_data_dtype(np.int16)
        header.set_data_shape(self.shape[:3])
        header.set_xyzt_units('mm')
        empty_img = np.zeros(self.shape[:3], dtype=np.int16)
        empty_nii = nib.Nifti1Image(empty_img, self.affine, header)
        return Nifti1Handler(empty_nii)
        
    def min(self):
        """
        Return the minimum value in the image data.
        """
        return self.img.dataobj.min()

    def max(self):
        """
        Return the maximum value in the image data.
        """
        return self.img.dataobj.max()
                      
    def extract_volume(self, vol_index=0):
        """
        Extract a specific volume from a 4D NIfTI image.
        If the image is already 3D, returns the original image.
        
        Parameters:
            vol_index (int): The index of the volume to extract (default is 0).
        
        Returns:
            Nifti1Handler: A new Nifti1Handler object containing the extracted 3D volume.
        """
        return Nifti1Handler(extract_volume_with_header(self.img, vol_index))
    
    def truncate_volumes(self, start=0, end=None, step=1):
        return Nifti1Handler(truncate_volumes(self.img, start, end, step))
    
    def to_canonical(self):
        """
        Convert the image to canonical (RAS+) orientation.
        This method uses nibabel.as_closest_canonical to reorient the image.
        
        Returns:
            Nifti1Handler: A new Nifti1Handler object in canonical (RAS+) orientation.
        """
        can_img = fast_to_canonical(self.img)
        return Nifti1Handler(can_img)
    
    def crop(self, voxel_min, voxel_max):
        """
        Crop the image in voxel space while preserving world coordinates.
        
        The cropped image will have its affine adjusted so that voxel (0,0,0)
        corresponds to the world coordinate of the original voxel at voxel_min.
        
        Parameters:
            voxel_min (tuple of ints): The minimum voxel indices (inclusive) for cropping.
            voxel_max (tuple of ints): The maximum voxel indices (exclusive) for cropping.
        
        Returns:
            Nifti1Handler: A new Nifti1Handler object containing the cropped image.
        """
        shape = self.shape[:3]
        if any(vm < 0 or vm >= s for vm, s in zip(voxel_min, shape)):
            raise IndexError(f"voxel_min {voxel_min} is out of range for image shape {shape}.")
        if any(vm < 1 or vm > s for vm, s in zip(voxel_max, shape)):
            raise IndexError(f"voxel_max {voxel_max} is out of range for image shape {shape}.")
        return Nifti1Handler(crop_nifti_image(self.img, voxel_min, voxel_max))
    
    def resample(self, new_voxel_sizes, order=1):
        """
        Resample the image to new voxel sizes while preserving world coordinates.
        
        The function computes a new affine that keeps the same origin (world coordinate
        of voxel (0,0,0)) and orientation, then uses scipy.ndimage.affine_transform to interpolate
        the image data.
        
        Parameters:
            new_voxel_sizes (tuple of 3 floats): Desired voxel sizes (in mm) for each spatial dimension.
            order (int): The interpolation order (default is 1 for linear interpolation).
            
        Returns:
            Nifti1Handler: A new Nifti1Handler object with the resampled image.
        """
        return Nifti1Handler(resample_nifti_image(self.img, new_voxel_sizes, order))
    
    def noncanonical_to_canonical_index(self, noncan_index):
        """
        Convert a voxel index from the non-canonical (original) image space to the canonical (RAS+) space.
        
        Parameters:
            noncan_index (tuple of ints): Voxel index in the non-canonical image space.
        
        Returns:
            tuple of ints: The corresponding voxel index in canonical (RAS+) space.
        """
        shape = self.shape[:3]
        if any(idx < 0 or idx >= s for idx, s in zip(noncan_index, shape)):
            raise IndexError(f"Non-canonical index {noncan_index} is out of range for image shape {shape}.")
        return noncanonical_to_canonical_index(self.img, noncan_index)
    
    def canonical_to_noncanonical_index(self, can_index):
        """
        Convert a voxel index from the canonical (RAS+) image space to the non-canonical (original) image space.
        
        Parameters:
            can_index (tuple of ints): Voxel index in the canonical (RAS+) image space.
        
        Returns:
            tuple of ints: The corresponding voxel index in the non-canonical image space.
        """
        # Convert and then verify that the resulting non-canonical index is within bounds.
        noncan_index = canonical_to_noncanonical_index(self.img, can_index)
        shape = self.shape[:3]
        if any(idx < 0 or idx >= s for idx, s in zip(noncan_index, shape)):
            raise IndexError(f"Canonical index {can_index} converts to out-of-range non-canonical index {noncan_index} for image shape {shape}.")
        return noncan_index
    
    def index_to_coord(self, *indices):
        """
        Convert voxel indices to world coordinates.
        
        Parameters:
            *indices (tuple of ints): Voxel indices in the image space.
        
        Returns:
            tuple of floats: The corresponding world coordinates.
        """
        shape = self.shape[:3]
        if len(indices) < 3:
            raise ValueError("At least 3 indices are required.")
        for idx, s in zip(indices[:3], shape):
            if idx < 0 or idx >= s:
                raise IndexError(f"Index {idx} is out of range for dimension size {s}.")
        return index_to_coord(self.img.affine, *indices)
    
    def coord_to_index(self, *coords):
        """
        Convert world coordinates to voxel indices.
        
        Parameters:
            *coords (tuple of floats): World coordinates.
        
        Returns:
            tuple of ints: The corresponding voxel indices.
        """
        indices = coord_to_index(self.img.affine, *coords)
        shape = self.shape[:3]
        for idx, s in zip(indices, shape):
            if idx < 0 or idx >= s:
                raise ValueError(f"Coordinate {coords} converts to index {indices}, which is out of range for shape {shape}.")
        return indices
    
    def save(self, filename):
        """
        Save the current NIfTI image to a file.
        
        Parameters:
            filename (str): The filename (including path) where the image will be saved.
        """
        nib.save(self.img, filename)
    
    def get_nibobj(self):
        """
        Return the underlying nibabel NIfTI image.
        
        Returns:
            nibabel.Nifti1Image: The current NIfTI image.
        """
        return self.img

    @staticmethod
    def _round_affine(affine, decimal_precision):
        """
        Round the affine matrix to the specified decimal precision.
        
        Parameters:
            decimal_precision (int): The number of decimal places to round to.
        
        Returns:
            np.ndarray: The rounded affine matrix.
        """
        if isinstance(decimal_precision, int) and decimal_precision >= 0:
            return np.round(affine, decimal_precision)
        else:
            raise ValueError("decimal_precision must be a non-negative integer.")

    def canonical_to(self, target_handler, decimal_precision=None):
        """
        Aligns the canonical (RAS+) image to the orientation and affine of the target image.
        (For example, used when aligning a mask drawn on a to_canonical processed image back 
        to the original image's orientation/affine.)

        Parameters:
            target_handler (Nifti1Handler): Object containing the original image information.
            decimal_precision (int, optional): Number of decimal places to round the affine matrix.

        Returns:
            Nifti1Handler: Image realigned to match the orientation and affine of the target.
        """
        target_affine = target_handler.affine.copy()
        target_ornt = nib.orientations.io_orientation(target_affine)
        current_ornt = nib.orientations.io_orientation(self.affine)

        transform = nib.orientations.ornt_transform(current_ornt, target_ornt)
        reoriented_data = nib.orientations.apply_orientation(self.img.get_fdata(), transform)
        
        new_affine = target_affine.copy()
        if decimal_precision:
            new_affine = np.round(new_affine, decimals=decimal_precision)
        
        new_img = nib.Nifti1Image(reoriented_data, new_affine, self.img.header)
        return Nifti1Handler(new_img)

    def get_meshgrid(self):
        """
        Generate a meshgrid of voxel indices for the image.
        
        Returns:
            tuple of np.ndarray: Meshgrid arrays for each dimension of the image.
        """
        return create_coordinate_meshgrid(self.img.shape[:3], self.affine)
    
    def reslice_to(self, target_handler, order=1):
        """
        Reslice the current image to match the coordinate system of the target image.
        The world coordinates are preserved.
        
        Parameters:
            target_handler (Nifti1Handler): The target image with the coordinate system to reslice.
            order (int): The interpolation order (default is 1 for linear interpolation).
        
        Returns:
            Nifti1Handler: A new Nifti1Handler object containing the resliced image.
        """
        # Get the affine and shape of the target image
        target_affine = target_handler.affine.copy()
        target_shape = target_handler.shape[:3]
        
        # Get the data of the current image
        current_data = self.img.get_fdata()
        
        # Calculate the world coordinates of the target grid
        i, j, k = np.meshgrid(
            np.arange(target_shape[0]),
            np.arange(target_shape[1]),
            np.arange(target_shape[2]),
            indexing='ij'
        )
        grid = np.vstack([i.ravel(), j.ravel(), k.ravel(), np.ones_like(i.ravel())])
        
        # Calculate the world coordinates of the target grid
        world_coords = np.dot(target_affine, grid)
        
        # Calculate the indices of the current image
        inv_affine = np.linalg.inv(self.affine)
        voxel_coords = np.dot(inv_affine, world_coords)
        
        # Prepare the coordinates for interpolation
        voxel_coords = voxel_coords[:3, :]
        
        # Check if the coordinates are within the image range
        x_valid = (voxel_coords[0] >= 0) & (voxel_coords[0] < current_data.shape[0]-1)
        y_valid = (voxel_coords[1] >= 0) & (voxel_coords[1] < current_data.shape[1]-1)
        z_valid = (voxel_coords[2] >= 0) & (voxel_coords[2] < current_data.shape[2]-1)
        valid = x_valid & y_valid & z_valid
        
        # Create a new image data
        new_data = np.zeros(target_shape)
        valid_coords = voxel_coords[:, valid]
        new_data_flat = map_coordinates(current_data, valid_coords, order=order)
        
        # Convert the flat data to the original shape
        new_data_raveled = np.zeros(np.prod(target_shape))
        new_data_raveled[valid] = new_data_flat
        new_data = new_data_raveled.reshape(target_shape)
        
        # Create a new NIfTI image
        new_img = nib.Nifti1Image(new_data, target_affine, self.img.header)
        return Nifti1Handler(new_img)
    
    def is_oblique(self, tol=1e-3):
        """
        Determine whether the affine matrix is oblique.
        
        Parameters:
            tol (float): Tolerance for comparing the original affine's 3x3 part 
                        with the reconstructed major orientation.
        
        Returns:
            bool: True if the affine is oblique (i.e. has small rotations or shear 
                beyond the major axis alignment), False otherwise.
        """
        # Copy the original affine matrix and extract the 3x3 part
        orig_affine = self.affine.copy()
        R = orig_affine[:3, :3]
        
        # Calculate the scale of each axis
        scales = np.linalg.norm(R, axis=0)
        
        # Use nibabel to determine the major orientation (primary axes)
        ornt = nib.orientations.io_orientation(orig_affine)
        
        # Create a rotation matrix that retains only the major orientation
        major_rot = np.zeros((3, 3))
        for i, (ax, flip) in enumerate(ornt):
            major_rot[int(ax), i] = 1 if flip >= 0 else -1
        # Apply the scale
        major_rot = major_rot * scales
        
        # If the original rotation/scale matrix R and major_rot are identical within tol, it is not oblique
        return not np.allclose(R, major_rot, atol=tol)
    
    def deoblique(self, warp_to_worldspace=False, decimal_precision=3):
        """
        Remove oblique elements from the affine matrix.
        
        Parameters:
            warp_to_worldspace (bool): 
                - False: Only modify the affine to remove obliqueness (i.e. keep only major axis directions).
                - True: Warp (reslice) the image data so that the oblique rotation is applied.
        
        Returns:
            Nifti1Handler: A new Nifti1Handler object with the deobliqued image.
        """
        # Extract the original affine, origin, and scale of each axis
        orig_affine = self.affine.copy()
        origin = orig_affine[:3, 3].copy()
        scale = np.sqrt(np.sum(orig_affine[:3, :3] ** 2, axis=0))
        
        # Use nibabel to determine the major orientation (major axis)
        ornt = nib.orientations.io_orientation(orig_affine)
        # Create a rotation matrix that keeps only the major orientation for each axis
        new_rot = np.zeros((3, 3))
        for i, (ax, flip) in enumerate(ornt):
            new_rot[int(ax), i] = 1 if flip >= 0 else -1
        # Apply scale
        new_rot = new_rot * scale
        
        # Prepare the new affine matrix
        new_affine = np.eye(4)
        new_affine[:3, :3] = new_rot
        new_affine[:3, 3] = origin
        # Round the affine to the specified decimal precision
        new_affine = self._round_affine(new_affine, decimal_precision) if decimal_precision else new_affine

        if not warp_to_worldspace:
            # Simply update the affine without reslicing the data
            new_img = nib.Nifti1Image(self.img.get_fdata(), new_affine, self.img.header)
            return Nifti1Handler(new_img)
        else:
            # warp_to_worldspace=True: Reslice the image data to match the new affine
            shape = self.img.shape[:3]
            # Generate a grid for the target image space
            i, j, k = np.meshgrid(
                np.arange(shape[0]),
                np.arange(shape[1]),
                np.arange(shape[2]),
                indexing='ij'
            )
            grid = np.vstack([i.ravel(), j.ravel(), k.ravel(), np.ones(i.size)])
            
            # Calculate world coordinates using the new affine
            world_coords = new_affine.dot(grid)
            # Use the inverse of the original affine to transform to the original voxel coordinates
            inv_orig_affine = np.linalg.inv(orig_affine)
            orig_vox_coords = inv_orig_affine.dot(world_coords)[:3, :]
            
            # Reslice using map_coordinates (linear interpolation)
            resliced_data = map_coordinates(
                self.img.get_fdata(), orig_vox_coords, order=1, mode='nearest'
            ).reshape(shape)
            
            new_img = nib.Nifti1Image(resliced_data, new_affine, self.img.header)
            return Nifti1Handler(new_img)