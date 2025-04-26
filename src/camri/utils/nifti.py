import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Union
from nibabel.orientations import io_orientation, ornt_transform, inv_ornt_aff, apply_orientation
from scipy.ndimage import affine_transform


def validate_input_file(input_file: Union[str, Path, nib.Nifti1Image]) -> nib.Nifti1Image:
    """ 
    """
    if isinstance(input_file, (str, Path)):
        input_file = Path(input_file) if isinstance(input_file, str) else input_file
        try:
            if input_file.is_file():
                return nib.load(input_file)
            else:
                raise FileNotFoundError(f"File not found: {input_file}")
        except Exception as e:
            raise ValueError(f"Failed to load the file. Error: {e}")
    elif isinstance(input_file, nib.Nifti1Image):
        return input_file
    else:
        raise TypeError("Invalid input: Expected a file path (str) or a nibabel.Nifti1Image object.")


def fast_to_canonical(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Quickly reorient a NIfTI image to canonical (RAS+) orientation using axis transforms and flips.

    This function computes the reorientation transformation (a combination of axis permutation
    and sign-flipping) from the image's original orientation to the canonical orientation (i.e.
    axes ordered as 0, 1, 2 with positive directions). It then applies this transformation to the
    image data and updates the affine matrix accordingly.

    Parameters:
        img (nibabel.Nifti1Image): The input (non-canonical) NIfTI image.

    Returns:
        nibabel.Nifti1Image: A new NIfTI image reoriented to canonical (RAS+) space.
    """
    orig_affine = img.affine
    # Get the data. For very large images, consider using img.dataobj to avoid loading everything.
    data = img.get_fdata()
    
    # Get the current orientation of the image.
    orig_ornt = io_orientation(orig_affine)
    # Define the canonical orientation: axes 0,1,2 in increasing order with positive direction.
    canonical_ornt = np.array([[0, 1],
                               [1, 1],
                               [2, 1]])
    # Compute the transform (a 3x2 array) that maps from the original to canonical orientation.
    transform = ornt_transform(orig_ornt, canonical_ornt)
    
    # Apply the orientation transform to the data.
    new_data = apply_orientation(data, transform)
    
    # Compute the new affine with the correct multiplication order.
    new_affine = orig_affine.dot(inv_ornt_aff(transform, img.shape[:3]))
    
    # compose nifti object
    new_img = compose_nifti(new_data, img, affine=new_affine)
    return new_img


def crop_nifti_image(img: nib.Nifti1Image, voxel_min: tuple[int, int, int], voxel_max: tuple[int, int, int]) -> nib.Nifti1Image:
    """
    Crop a NIfTI image (3D or 4D) in voxel space while preserving world coordinates.
    
    The function extracts the region specified by voxel indices [voxel_min, voxel_max)
    and updates the affine matrix so that the new image's (0,0,0) voxel corresponds to
    the same world coordinate as the original image's voxel at voxel_min.
    
    Parameters:
        img (nibabel.Nifti1Image): Input NIfTI image.
        voxel_min (tuple of ints): Minimum voxel indices (inclusive) in each dimension (e.g., (i_min, j_min, k_min)).
        voxel_max (tuple of ints): Maximum voxel indices (exclusive) in each dimension (e.g., (i_max, j_max, k_max)).
        
    Returns:
        nibabel.Nifti1Image: The cropped NIfTI image with updated affine.
    """
    # Slice the data (works for 3D or 4D; if 4D, cropping is applied to the first 3 dims)
    data = img.get_fdata()
    if data.ndim == 3:
        cropped_data = data[voxel_min[0]:voxel_max[0],
                            voxel_min[1]:voxel_max[1],
                            voxel_min[2]:voxel_max[2]]
    elif data.ndim >= 4:
        # Apply cropping to the spatial dimensions only
        cropped_data = data[voxel_min[0]:voxel_max[0],
                            voxel_min[1]:voxel_max[1],
                            voxel_min[2]:voxel_max[2], ...]
    else:
        raise ValueError("Image data must be at least 3D.")
    
    # Compute the new origin: world coordinate corresponding to voxel_min in the original image.
    new_origin = nib.affines.apply_affine(img.affine, voxel_min)
    
    # Create a new affine that preserves the rotation/scaling but updates the translation.
    new_affine = img.affine.copy()
    new_affine[:3, 3] = new_origin

    # compose nifti object
    new_img = compose_nifti(cropped_data, img, affine=new_affine)
    return new_img


def resample_nifti_image(
    img: nib.Nifti1Image, 
    new_voxel_sizes: tuple[float, float, float], 
    order: int = 1
    ) -> nib.Nifti1Image:
    """
    Resample a NIfTI image to new voxel sizes while preserving world coordinates.
    
    This function computes a new affine matrix with the desired voxel sizes while keeping
    the world coordinate of the origin (voxel (0,0,0)) unchanged. It then uses
    scipy.ndimage.affine_transform to interpolate the image data.
    
    Parameters:
        img (nibabel.Nifti1Image): Input NIfTI image.
        new_voxel_sizes (tuple of 3 floats): Desired voxel sizes (in mm) for each spatial dimension.
        order (int): The interpolation order (default is 1 for linear interpolation).
        
    Returns:
        nibabel.Nifti1Image: The resampled NIfTI image with updated affine.
    """
    # Get the current affine and compute the old voxel sizes from the first 3 columns.
    old_affine = img.affine
    old_voxel_sizes = np.sqrt(np.sum(old_affine[:3, :3] ** 2, axis=0))
    
    # Compute scale factors (how many times more voxels will be needed)
    scale_factors = np.array(old_voxel_sizes) / np.array(new_voxel_sizes)
    
    # Determine the new shape (rounding up to preserve FOV)
    old_shape = np.array(img.shape[:3])
    new_shape = np.ceil(old_shape * scale_factors).astype(int)
    
    # Compute new affine:
    #   1. Extract the direction cosines (columns normalized)
    directions = old_affine[:3, :3] / old_voxel_sizes
    #   2. Multiply by new voxel sizes to get new scaling while keeping the same orientation
    new_affine = np.zeros((4, 4))
    new_affine[:3, :3] = directions * new_voxel_sizes
    # Keep the same origin (world coordinate of voxel (0,0,0))
    new_affine[:3, 3] = old_affine[:3, 3]
    new_affine[3, 3] = 1.0
    
    # Compute the transformation matrix from new voxel space to original voxel space.
    # We have: world = old_affine @ voxel_orig and world = new_affine @ voxel_new.
    # Therefore, voxel_orig = inv(old_affine) @ new_affine @ voxel_new.
    transform = np.linalg.inv(old_affine) @ new_affine
    
    # Apply affine transformation using scipy.ndimage.affine_transform.
    # Note: affine_transform maps output voxel coordinate to input voxel coordinate.
    resampled_data = affine_transform(
        img.get_fdata(),
        matrix=transform[:3, :3],
        offset=transform[:3, 3],
        output_shape=new_shape,
        order=order
    )
    
    # compose nifti object
    new_img = compose_nifti(resampled_data, img, affine=new_affine)
    return new_img


def extract_volume_with_header(
        img4d: nib.Nifti1Image, 
        vol_index: int = 0
    ) -> nib.Nifti1Image:
    """
    Extract a specific volume from a 4D NIfTI image and convert it to a 3D image while preserving header information 
    (e.g., sform, qform). The function adjusts the header's dimension information to match the 3D data, removing 
    only the fourth axis information.

    Parameters:
        img4d (nibabel.Nifti1Image): The input 4D NIfTI image.
        vol_index (int): The index of the volume to extract (default is 0).

    Returns:
        nibabel.Nifti1Image: A new 3D NIfTI image object with the updated header.
    """
    if img4d.ndim < 4:
        return img4d
    # Extract the desired volume without loading the entire 4D data into memory
    data_3d = img4d.dataobj[..., vol_index]

    # compose nifti object
    img3d = compose_nifti(data_3d, img4d)
    return img3d


def truncate_volumes(
        img4d: nib.Nifti1Image, 
        start: int = 0, 
        end: int | None = None, 
        step: int = 1
    ) -> nib.Nifti1Image:
    """
    Extract a subset of volumes from a 4D NIfTI image while preserving header information
    (e.g., sform, qform). The header's dimension information is updated to match the output data.
    
    Parameters:
        img4d (nibabel.Nifti1Image): The input 4D NIfTI image.
        start (int): The starting volume index (default is 0).
        end (int or None): The ending volume index (exclusive). If None, extraction goes to the end.
        step (int): The step between volumes (default is 1).
    
    Returns:
        nibabel.Nifti1Image: A new NIfTI image object containing the extracted volume(s) with updated header.
                             (The output image will be 3D if a single volume is selected, or 4D if multiple volumes are selected.)
    """
    # Create a new NIfTI image with the subset data and the original affine.
    if img4d.ndim < 4:
        return img4d
    data_subset = img4d.dataobj[..., slice(start, end, step)]
    new_img = compose_nifti(data_subset, img4d)
    return new_img


def clear_negzero(
            array: np.ndarray,
        ) -> np.ndarray:
    zero_mask = np.nonzero(array == 0)
    array[zero_mask] = abs(0)
    return array


def compose_nifti(
        data: np.ndarray, 
        template_nifti: nib.Nifti1Image, 
        affine: np.ndarray | None = None, 
        mask_idx: tuple[int, int, int] | None = None
    ) -> nib.Nifti1Image:
    """
    Compose a NIfTI object using a 3D or 4D array + affine, or a 2D array + mask_idx.
    The template_nifti is required to template the header information.
    """
    hdr = template_nifti.header.copy()
    if mask_idx:
        # only use this if image has been masked
        if len(data.shape) > 1 and len(data.shape) < 3:
            dataobj = np.zeros(list(template_nifti.shape[:3]) + [data.shape[-1]])
            dataobj[mask_idx] = data.copy()
        else:
            dataobj = np.zeros(data.shape)
            dataobj[mask_idx] = data.copy()[mask_idx]
    else:
        if len(data.shape) < 3:
            raise ValueError("data must be at least 3D without mask")
        dataobj = data.copy()
    
    # header
    hdr = template_nifti.header.copy()
    hdr.set_data_shape(dataobj.shape)
    
    # affine
    if not isinstance(affine, np.ndarray):
        affine = template_nifti.affine.copy()
    affine = clear_negzero(np.nextafter(affine, np.ceil(affine)))
    # compose nifti object
    nifti = nib.Nifti1Image(dataobj, affine, header=hdr)
    return nifti


def noncanonical_to_canonical_index(noncan_index, img):
    """
    Convert a voxel index from the non-canonical (original) image space to the canonical (RAS+) image space.
    
    Parameters:
        noncan_index (tuple or list of 3 ints): Voxel index in the non-canonical image space.
        img (nibabel.Nifti1Image): The non-canonical NIfTI image.
        
    Returns:
        tuple of 3 ints: The corresponding voxel index in canonical (RAS+) space.
    """
    # Ensure the index is a numpy array
    noncan_index = np.array(noncan_index)
    
    # Compute the orientation of the original image
    orig_ornt = io_orientation(img.affine)
    # Define canonical orientation: (axis 0,1,2 all positive direction)
    canonical_ornt = np.array([[0, 1],
                               [1, 1],
                               [2, 1]])
    
    # Compute the transformation from the original orientation to canonical orientation
    T = ornt_transform(orig_ornt, canonical_ornt)
    
    # The size (shape) of the original image along each axis
    orig_shape = np.array(img.shape[:3])
    # The size along each canonical axis is given by the corresponding non-canonical axis size:
    canonical_shape = np.array([orig_shape[int(T[i, 0])] for i in range(3)])
    
    # Compute the canonical index
    can_index = np.zeros(3, dtype=int)
    for i in range(3):
        src = int(T[i, 0])
        flip = T[i, 1]
        if flip == 1:
            can_index[i] = noncan_index[src]
        else:
            # When flipping, the index must be reversed within that axis
            can_index[i] = canonical_shape[i] - 1 - noncan_index[src]
    return tuple(can_index.tolist())


def canonical_to_noncanonical_index(can_index, img):
    """
    Convert a voxel index from the canonical (RAS+) image space to the non-canonical (original) image space.
    
    Parameters:
        can_index (tuple or list of 3 ints): Voxel index in the canonical (RAS+) image space.
        img (nibabel.Nifti1Image): The non-canonical NIfTI image.
        
    Returns:
        tuple of 3 ints: The corresponding voxel index in the non-canonical image space.
    """
    # Ensure the canonical index is a numpy array
    can_index = np.array(can_index)
    
    # Get the original orientation of the image
    orig_ornt = io_orientation(img.affine)
    canonical_ornt = np.array([[0, 1],
                               [1, 1],
                               [2, 1]])
    # Compute the transformation from canonical to original orientation
    T_inv = ornt_transform(canonical_ornt, orig_ornt)
    
    # For computing the inverse mapping, we need the canonical shape.
    # This is the same as in the previous function:
    orig_shape = np.array(img.shape[:3])
    T = ornt_transform(orig_ornt, canonical_ornt)
    canonical_shape = np.array([orig_shape[int(T[i, 0])] for i in range(3)])
    
    # Initialize an array for the non-canonical index.
    noncan_index = np.zeros(3, dtype=int)
    # T_inv maps canonical axes to non-canonical axes.
    for i in range(3):
        src = int(T_inv[i, 0])  # the non-canonical axis corresponding to canonical axis i
        flip = T_inv[i, 1]
        if flip == 1:
            noncan_index[src] = can_index[i]
        else:
            noncan_index[src] = orig_shape[src] - 1 - can_index[i]
    return tuple(noncan_index.tolist())


def index_to_coord(affine, i, j, k, decimals=3):
    """
    Vox
    el index (i, j, k) → physical coordinate (x, y, z)
    """
    indices = np.atleast_2d([i, j, k])
    indices_h = np.hstack([indices, np.ones((indices.shape[0], 1))])
    coords = indices_h @ affine.T
    return tuple(np.round(coords[0, :3], decimals=decimals).tolist())


def coord_to_index(affine, x, y, z):
    """
    Physical coordinate (x, y, z) → voxel index (i, j, k)
    """
    coords = np.atleast_2d([x, y, z])
    inv_affine = np.linalg.inv(affine)
    coords_h = np.hstack([coords, np.ones((coords.shape[0], 1))])
    indices = coords_h @ inv_affine.T
    return tuple(np.round(indices[0, :3]).astype(int).tolist())


def create_coordinate_meshgrid(shape: tuple[int, int, int], affine: np.ndarray) -> np.ndarray:
    """
    Generate a 3D coordinate meshgrid in world coordinates based on the given shape and affine transformation.

    This function creates a meshgrid of voxel indices for a 3D volume with the specified shape, 
    applies the affine transformation to convert the voxel indices to world coordinates, 
    and returns the resulting 3D world coordinate grid.

    Args:
        shape (tuple[int, int, int]): The shape of the 3D volume (e.g., (x, y, z)).
        affine (np.ndarray): A 4x4 affine transformation matrix that maps voxel indices 
                             to world coordinates.

    Returns:
        np.ndarray: A 3D array of shape (x, y, z, 3), where each element contains the 
                    world coordinates (x, y, z) corresponding to the voxel at that position.
    """
    ijk = np.indices(shape).reshape(3, -1).T
    world_coords = nib.affines.apply_affine(affine, ijk).reshape(shape + (3,))
    return world_coords
