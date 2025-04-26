import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
from ..utils.nifti import compose_nifti
    

def skull_stripping(
        input_nii: nib.Nifti1Image, 
        mask_nii: nib.Nifti1Image
    ) -> nib.Nifti1Image:
    """
    Apply skull stripping to a NIfTI image using a binary mask.
    
    Parameters
    ----------
    input_nii : nibabel.Nifti1Image
        Input NIfTI image to be skull stripped
    mask_nii : nibabel.Nifti1Image  
        Binary mask NIfTI image where non-zero values indicate brain tissue
        
    Returns
    -------
    nibabel.Nifti1Image
        Skull stripped NIfTI image containing only voxels within the mask
    """
    mask_idx = np.nonzero(mask_nii.dataobj)
    data = np.asarray(input_nii.dataobj)[mask_idx]
    return compose_nifti(data, input_nii, mask_idx=mask_idx)


def estimate_sigma(dxyz: tuple[float, float, float], fwhm: float) -> float:
    """
    Calculate sigma values for Gaussian smoothing from FWHM and voxel dimensions.
    
    Parameters
    ----------
    dxyz : tuple[float, float, float]
        Voxel dimensions in x, y, z directions
    fwhm : float
        Full-width at half-maximum in mm
        
    Returns
    -------
    ndarray
        Sigma values for each dimension scaled by voxel size
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return sigma / np.array(dxyz)


def spatial_smoothing(input_nii: nib.Nifti1Image, fwhm: float, mask_nii: nib.Nifti1Image | None = None) -> nib.Nifti1Image:
    dxyz = input_nii.header['pixdim'][1:4]
    dataobj = input_nii.get_fdata()
    
    has_mask = isinstance(mask_nii, nib.Nifti1Image)
    if has_mask:
        mask_bool = np.asarray(mask_nii.dataobj) == 0
        dataobj[mask_bool] = 0
    else:
        mask_bool = None
    
    sigma = estimate_sigma(dxyz, fwhm)
    if len(input_nii.shape) == 3:
        smoothed_dataobj = gaussian_filter(dataobj, sigma).astype(float)
    else:
        smoothed_dataobj = []
        for t in range(dataobj.shape[-1]):
            smoothed_dataobj.append(gaussian_filter(dataobj[..., t], sigma).astype(float))
        smoothed_dataobj = np.stack(smoothed_dataobj, axis=-1)
    
    if has_mask:
        smoothed_dataobj[mask_bool] = 0
    return compose_nifti(smoothed_dataobj, input_nii)

