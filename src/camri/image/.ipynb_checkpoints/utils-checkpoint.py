import nibabel as nib
import numpy as np
from pathlib import Path


def validate_nifti_input(nibobj):
    if isinstance(nibobj, str):
        try:
            if Path(nibobj).is_file():
                return nib.load(nibobj)
            else:
                raise FileNotFoundError(f"File not found: {nibobj}")
        except Exception as e:
            raise ValueError(f"Failed to load the file. Error: {e}")
    elif isinstance(nibobj, nib.Nifti1Image):
        return nibobj
    else:
        raise TypeError("Invalid input: Expected a file path (str) or a nibabel.Nifti1Image object.")


def compose_nifti(data, tempalte_nifti, affine=None, affine_decimals=3, mask_idx=None):
    """ compose nifti object using 3D or 4D array + affine, or 2D array + mask_idx,
    template_nifti required for templating header information
    """
    if mask_idx:
        # only use this if image has been masked
        if len(data.shape) > 1:
            dataobj = np.zeros(list(tempalte_nifti.shape[:3]) + [data.shape[-1]])
        else:
            dataobj = np.zeros(tempalte_nifti.shape[:3])
        dataobj[mask_idx] = data.copy()
    else:
        dataobj = data.copy()
        
    if not isinstance(affine, np.ndarray):
        affine = tempalte_nifti.affine.copy()
    affine = clear_negzero(np.round(affine, decimals=affine_decimals))
    nifti = nib.Nifti1Image(dataobj, affine)
    if qform_code := tempalte_nifti.header['qform_code'].tolist():
        nifti.set_qform(affine, qform_code)
    if sform_code := tempalte_nifti.header['sform_code'].tolist():
        nifti.set_sform(affine, sform_code)
    return nifti


def clear_negzero(array):
    zero_mask = np.nonzero(array == 0)
    array[zero_mask] = abs(0)
    return array
    
    
def split_affine(affine, decimals=3):
    affine = np.round(clear_negzero(affine.copy()), 
                      decimals=decimals)
    rotate = affine[:3, :3]
    origin = affine[:3, 3]
    return rotate, origin


def merge_affine(rotate, origin, decimals=3):
    affine = np.eye(4)
    affine[:3, :3] = rotate
    affine[:3, 3] = origin
    return clear_negzero(np.round(affine, 
                                  decimals=decimals))


def rotate_function(array, radian, axis=0, decimals=3):
    """
    array = 3x3 or 3 elements
    """
    np.set_printoptions(precision=3, suppress=True)
    if axis < 0 and axis > -3:
        axis += 3
    rotate_function = {
        0: np.array([[np.cos(radian), -np.sin(radian), 0], 
                     [np.sin(radian), np.cos(radian), 0], 
                     [0, 0, 1]]),
        1: np.array([[np.cos(radian), 0, np.sin(radian)], 
                     [0, 1, 0], 
                     [-np.sin(radian), 0, np.cos(radian)]]),
        2: np.array([[1, 0, 0],
                     [0, np.cos(radian), -np.sin(radian)],
                     [0, np.sin(radian), np.cos(radian)]])
    }
    return clear_negzero(np.round(rotate_function[axis].dot(array), 
                         decimals=decimals))


def rotate_affine(affine, gamma=0, phi=0, theta=0):
    """
    gamma: rotate radian for 0 axis
    phi(fee): rotate radian for 1 axis
    theta: rotate radian for 2 axis
    """
    rotate, origin = aff2mat(affine)
    for i, radian in enumerate([gamma, phi, theta]):
        rotate = rotate_function(rotate, radian, axis=i)
        origin = rotate_function(origin, radian, axis=1)
    return merge_affine(rotate, origin)


def remove_element_at(array, index):
    if isinstance(array, np.ndarray):
        array = array.tolist()
        _ = array.pop(index)
    else:
        array = array[:index] + array[index+1:]
    return array