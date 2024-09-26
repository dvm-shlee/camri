import nibabel as nib
import numpy as np

def compose_nii(data, ref_nii, affine=None, mask_idx=None):
    """
    TODO: duplicate the header information will be useful
    """
    if mask_idx:
        # only use this if image has been masked
        if len(data.shape) > 1:
            dataobj = np.zeros(list(ref_nii.shape[:3]) + [data.shape[-1]])
        else:
            dataobj = np.zeros(ref_nii.shape[:3])
        dataobj[mask_idx] = data.copy()
    else:
        dataobj = data.copy()
    
    affine = affine or ref_nii.affine.copy()
    nibobj = nib.Nifti1Image(dataobj, affine)
    if qform_code := ref_nii.header['qform_code'].tolist():
        nibobj.set_qform(affine, qform_code)
    if sform_code := ref_nii.header['sform_code'].tolist():
        nibobj.set_sform(affine, sform_code)
    return nibobj


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
    
    
def aff2mat(affine, decimals=3):
    "for backward comp, deprecated"
    return split_affine(affine, decimals)


def mat2aff(rotate, origin, decimals=3):
    "for backward comp, deprecated"
    return merge_affine(rotate, origin, decimals)


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
    return mat2aff(rotate, origin)


def get_orient_code(rotate):
    """# integrated to OrientTool
    """
    direction = (clear_negzero(rotate / np.abs(rotate.sum(0))).sum(1) > 0).astype(int)
    orient_codes = {
        0: 'LR',
        1: 'PA',
        2: 'IS'
    }
    return [orient_codes[i][direction[i]] for i in np.nonzero(rotate)[1]]


def get_meshgrid(matrix_size, affine):
    """return meshgrid of image (x, y, z coordinate space based on affine)
    # integrated to OrientTool
    """
    rotation, origin = aff2mat(affine)
    resolution = rotation.sum(0)
    stop_xyz = origin + matrix_size * resolution
    meshgrid = {}
    for idx, axis in enumerate(['x', 'y', 'z']):
        meshgrid[axis] = np.linspace(origin[idx], stop_xyz[idx], matrix_size[idx], endpoint=False)
    return meshgrid, resolution


def get_image_fov(meshgrid, resolusion, decimals=3):
    edge = resolusion / 2 
    fov = []
    for vals in meshgrid.values():
        fov.append((vals[0] - edge[0], a[-1] + edge[1]))
    return np.round(fov, decimals=decimals)

def get_viewport_fov(fov):
    fov_size = np.abs(fov[:, 1] - fov[:, 0])
    size_diff = fov_size.max() - fov_size
    viewport_fov = fov.copy()
    viewport_fov[:, 0] -= size_diff/2
    viewport_fov[:, 1] += size_diff/2
    return viewport_fov


def get_target_coord(ref_meshgrid, target_meshgrid, itk_coord):
    """calculate itk coordinate (voxel indices) of target image based on coordinates of reference image
    !!the field of view of target image must bigger or equal to reference image!!
    
    Returns:
    - target_fov_slicer: indices for target image to match FOV with reference
    - target_itk_coord: calculated target coordinates
    """
    target_fov_slicer = []
    target_itk_coord = []
    for idx, axis in enumerate(['x', 'y', 'z']):
        ref_res = np.diff(ref_meshgrid[axis])[0]
        target_res = np.diff(target_meshgrid[axis])[0]

        ref_slice = itk_coord[idx]
        ref_start, ref_pos, ref_stop = ref_meshgrid[axis][[0, ref_slice, -1]]
        
        fov_start = np.argmin(np.abs(target_meshgrid[axis] - (ref_start - ref_res/2 + target_res/2)))          
        fov_stop = np.argmin(np.abs(target_meshgrid[axis] - (ref_stop + ref_res/2 - target_res/2))) + 1
        
        target_fov_slicer.append(slice(fov_start, fov_stop))
        target_itk_coord.append(np.argmin(np.abs(target_meshgrid[axis] - ref_pos)))
    return tuple(target_fov_slicer), target_itk_coord


def get_ortho_slicer(fov_slicer, itk_coord):
    return {'sagital':(itk_coord[0], fov_slicer[1], fov_slicer[2]), 
            'coronal':(fov_slicer[0], itk_coord[1], fov_slicer[2]),
            'axial':(fov_slicer[0], fov_slicer[1], itk_coord[2])}


def get_transform(reference_affine, move_affine):
    """ #integrated to OrientTool """
    return reference_affine.dot(np.linalg.inv(move_affine))


def get_indice_from_coord(meshgrid, xyz_coord):
    """convert eucledian coordinate into numpy indices
    # integrated to OrientTool
    """
    itk_indice = []
    for i, axis in enumerate(['x', 'y', 'z']):
        itk_indice.append(np.argmin(np.abs(meshgrid[axis] - xyz_coord[i])))
    return itk_indice

    
def get_coord_from_indice(meshgrid, itk_indice):
    """convert numpy indices into eucledian coordinate
    # integrated into OrientTool
    """
    xyz_coord = []
    for i, axis in enumerate(['x', 'y', 'z']):
        grid = meshgrid[axis]
        coord = grid[itk_indice[i]] + np.diff(grid)[0]/2
        xyz_coord.append(np.round(coord, decimals=3))
    return xyz_coord