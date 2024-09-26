import numpy as np
import nibabel as nib
from ..image.utils import compose_nifti
    

def skull_stripping(input_nii, mask_nii):
    mask_idx = np.nonzero(mask_nii.dataobj)
    data = np.asarray(input_nii.dataobj)[mask_idx]
    return compose_nifti(data, input_nii, mask_idx=mask_idx)


def estimate_sigma(dxyz, fwhm: float) -> float:
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return sigma / np.array(dxyz)


def spatial_smoothing(input_nii, fwhm, mask_nii=None):
    from scipy.ndimage import gaussian_filter
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

