import numpy as np
import pandas as pd
import nibabel as nib
from .signal import signal_normalization
from .signal import nuisance_regression
from ..utils.nifti import compose_nifti


def load_volreg(path, mean_radius=9):
    """ Return motion parameter estimated from AFNI's 3dvolreg
    radian values will converted to distance based on given radius.

    :param path:        filepath of 1D data
    :param mean_radius: the distance from aural to central fissure of animal (default: 9mm for rat)
    :return:
    """

    def convert_radian2distance(volreg_, mean_radius_):
        volreg_[['Roll', 'Pitch', 'Yaw']] *= (np.pi / 180 * mean_radius_)
        return volreg_
    
    volreg = pd.read_csv(path, header=None, sep=r'\s+')
    volreg.columns = ['Roll', 'Pitch', 'Yaw', 'dI-S', 'dR-L', 'dA-P']
    r = np.round(np.sqrt(2) * mean_radius)
    return convert_radian2distance(volreg, r)


def trimming_dummies(nibobj, num_dummy=0):
    dataobj = np.asarray(nibobj.dataobj)[..., num_dummy:]
    return compose_nifti(dataobj, nibobj)


def trimming_mpfile(input_path, output_path, num_dummy=0):
    # for afni (just alternative function)
    volreg = load_volreg(input_path)
    with open(output_path, 'w') as f:
        for line in (volreg.loc[num_dummy:] - volreg.loc[num_dummy]).values:
            new_line = []
            for v in line.tolist():
                val, deci = str(np.round(v, decimals=4)).split('.')
                new_line.append(('.'.join([val, deci.zfill(4)])).rjust(8))
            f.write(' '.join(new_line) + '\n')


def prep_ort(ort_df, ort_timeshift=None, ort_square=None, ort_nii=None, **kwargs):
    """
    ort_square: bool - if True, add squared ort
    ort_timeshift: int - 0, 1, or 2 (if integer, time step performed
    crop: Optional, list[int, int], number of frame that will excluded (from first and last)
    """
    ort_df = ort_df.copy()
    if ort_timeshift:
        for shift in range(1, ort_timeshift+1):
            for col in ort_df.columns:
                if not col.endswith('t-1'):
                    new_col = f'{col}_t-{shift}'
                    rolled = np.roll(ort_df.loc[:, col], shift)
                    rolled[:shift] = 0
                    ort_df[new_col] = rolled
    if ort_square:
        for c in ort_df.columns:
            new_col = c + '^2'
            ort_df[new_col] = np.square(ort_df.loc[:, c])
    ort = ort_df.values.T
    if isinstance(ort_nii, nib.Nifti1Image):
        if 'img' not in kwargs.keys():
            raise TypeError
        img = kwargs['img']
        mask_ort = img[np.nonzero(ort_nii.dataobj)].mean(0)
        ort = np.concatenate([mask_ort[np.newaxis, :], ort], axis=0)
    return ort


def signal_processing(input_nii, mask_nii, 
                      port = 1,
                      ort_df = None, 
                      ort_timeshift = None, 
                      ort_square = None,
                      ort_nii = None, 
                      normalize_method='mode',
                      crop = None,
                      **kwargs):
    
    img = input_nii.get_fdata()
    if isinstance(ort_df, pd.DataFrame):
        ort = prep_ort(ort_df, ort_timeshift, ort_square, ort_nii, img=img)
    else:
        ort = None
    mask_idx = np.nonzero(mask_nii.dataobj)
    data = img[mask_idx]
    if normalize_method:
        if normalize_method == 'mode':
            kwargs.setdefault('mode', 1000)
            mode = kwargs.pop('mode')
            norm_data = signal_normalization(data, method=normalize_method, mode=mode)
        else:
            norm_data = signal_normalization(data, method=normalize_method)
    else:
        norm_data = data.copy()
    filt_data, nuis_data = nuisance_regression(norm_data, port, ort)
    filt_nii = compose_nifti(filt_data, input_nii, mask_idx=mask_idx)
    nuis_nii = compose_nifti(nuis_data, input_nii, mask_idx=mask_idx)
    return filt_nii, nuis_nii
