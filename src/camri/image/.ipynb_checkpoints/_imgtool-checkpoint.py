from tqdm.auto import tqdm
from ._utils import *

# final version 2024-09-11 shlee (transform)
## brkraw, scanner coordinate system, scanner's left->right, scanner's floor->ceiling, scanner bore (console->back))
### which correspond to subjects RSP+ for supine position for rodent
## neurology subject space should be RAS+ (subject's left->right, posterior->anterior, inferior->superior)

# RSP+: scanner orientation (for typical supine position) - qform_code = 1
# RAS+ for sform_code = 1


def get_resol(nibobj):
    return nibobj.header['pixdim'][1:4].copy()


def get_orient_matrix(nibobj):
    descale_tf = np.linalg.inv(np.diag(get_resol(nibobj)))
    return np.round(aff2mat(nibobj.affine)[0].dot(descale_tf), decimals=0)


def get_ras_affine(nibobj):
    org_res = get_resol(nibobj)
    orient_mat = get_orient_matrix(nibobj)
    img_res = np.abs(orient_mat.dot(org_res))
    rotate = np.diag(img_res)
    origin = -np.abs(aff2mat(nibobj.affine)[1])
    return mat2aff(rotate, origin)


def get_ras_transform(nibobj):
    """implement to orienttool"""
    orig_affine = nibobj.affine.copy()
    oc_affine = get_ras_affine(nibobj)
    return get_transform(oc_affine, orig_affine)


def get_ras_oriented_img(nibobj):
    """ function return image orientated RAS+ orthoginal with centered origin
    This is showing example of using affine transform to change orientation while keeping its resolution
    
    perform affine using scipy
    """
    from scipy.ndimage import affine_transform
    img = np.asarray(nibobj.dataobj)
    rotate, origin = aff2mat(get_ras_transform(nibobj))
    rotate /= np.abs(rotate.sum(0))  # if not normalize this, anisotropic image will be resampled
    
    orig_shape = nibobj.shape[:3]
    output_shape = np.abs(rotate.dot(orig_shape)).astype(int)
    shift = orig_shape * (rotate.sum(-1) < 0).astype(int)  # in scipy, the shift only required for flipped axis, its always rotate from center
    
    if len(img.shape) > 3:
        ras_img = []
        for frame_id in tqdm(range(img.shape[-1])):
            if len(img.shape) == 4:
                ras_img.append(affine_transform(img[..., frame_id], 
                                                rotate, shift, 
                                                output_shape=output_shape))
            elif len(img.shape) == 5 and img.shape[3] == 1:
                ras_img.append(affine_transform(img[..., frame_id].squeeze(),
                                                rotate, shift, 
                                                output_shape=output_shape)[..., np.newaxis])
            else:
                raise TypeError
        ras_img = np.stack(ras_img, axis=-1)
    else:
        ras_img = affine_transform(img, rotate, shift, output_shape=output_shape)
    return ras_img


def get_ras_img(nibobj):
    """
    This is more efficient way
    """
    dataobj = nibobj.get_fdata()
    orimat = get_orient_matrix(nibobj)
    _, target_order = np.nonzero(orimat)
    dim = len(dataobj.shape)
    if dim > target_order.shape[0]:
        target_order = np.append(target_order, dim-1)
    ras_dataobj = np.transpose(dataobj, target_order)
    slicer = tuple(slice(None, None, int(step)) for step in orimat.sum(1))
    ras_dataobj = ras_dataobj[slicer]
    return ras_dataobj


def reorient_to_ras(nibobj):
    """ function convert nibabel's nifti1image into orthogonal and centered origin image
    with the RAS+ oriented matrix
    """
    img = get_ras_img(nibobj)
    affine = get_ras_affine(nibobj)
    ras_nibobj = nib.Nifti1Image(img, affine)
    ras_nibobj.set_sform(affine, 1)
    return ras_nibobj


def crop_by_voxels(nibobj, l=0, r=0, p=0, a=0, i=0, s=0):
    img = np.asarray(nibobj.dataobj)
    shape = nibobj.shape
    img_res = get_resol(nibobj)
    rotate, origin = aff2mat(nibobj.affine)
    
    codes = get_orient_code(rotate)
    
    slices = []
    shift = []
    for j, c in enumerate(codes):
        if c == "L":
            l = shape[j] - l
            slices.append(slice(r, l))
            shift.append(r)
        elif c == "R":
            r = shape[j] - r
            slices.append(slice(l, r))
            shift.append(l)
        elif c == "P":
            p = shape[j] - p
            slices.append(slice(a, p))
            shift.append(a)
        elif c == "A":
            a = shape[j] - a
            slices.append(slice(p, a))
            shift.append(p)
        elif c == "I":
            i = shape[j] - i
            slices.append(slice(s, i))
            shift.append(s)
        else:
            s = shape[j] - s
            slices.append(slice(i, s))
            shift.append(i)
    orient_tf = get_orient_matrix(nibobj)
    origin += orient_tf.dot(np.array(shift) * img_res)
    cropped_img = img[tuple(slices)]
    affine = mat2aff(rotate, origin)
    cropped_nibobj = nib.Nifti1Image(cropped_img, affine)
    if qform_code := nibobj.header['qform_code'].tolist():
        cropped_nibobj.set_qform(affine, qform_code)
    if sform_code := nibobj.header['sform_code'].tolist():
        cropped_nibobj.set_sform(affine, sform_code)
    return cropped_nibobj


def gridding(nibobj):
    """return ras orientated image and its meshgrid"""
    img = get_ras_img(nibobj)
    affine = get_ras_affine(nibobj)
    meshgrid, resol = get_meshgrid(img.shape[:3], affine)
    return img, meshgrid, resol