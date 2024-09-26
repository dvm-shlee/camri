import sys
import numpy as np
import pandas as pd
from typing import Union


def tsnr(data: np.ndarray, io_handler=sys.stdout) -> Union[np.ndarray, int]:
    """ calculate temporal snr

    Args:
        data: 1d or 2d (V x T) data

    Returns:
        tsnr
    """
    io_handler.write('Calculating tSNR...')
    dim = data.shape
    if dim == 1:
        tsnr = data.mean() / data.std()
    else:
        mean = data.mean(-1)
        std = data.std(-1)
        tsnr = np.zeros(mean.shape)
        masked_mean = mean[np.nonzero(std)]
        masked_std = std[np.nonzero(std)]
        tsnr[np.nonzero(std)] = masked_mean / masked_std
    io_handler.write('[Done]\n')
    return tsnr


def dvars(data, io_handler=sys.stdout):
    io_handler.write('Calculating DVARS...')
    mask_idx = np.nonzero(data.mean(-1))
    diff_img = np.diff(data[mask_idx], axis=-1)
    dvars_ = np.sqrt(np.square(diff_img).mean(0))
    io_handler.write('[Done]\n')
    return np.insert(dvars_, 0, 0)


def bold_mean_std(data, io_handler=sys.stdout):
    io_handler.write('Calculating Mean and STD...')
    mask_idx = np.nonzero(data.mean(-1))
    masked_data = data[mask_idx]
    masked_data = (masked_data.T - masked_data.mean(1).T).T
    io_handler.write('[Done]\n')
    return masked_data.mean(0), masked_data.std(0)


def framewise_displacements(volreg, io_handler=sys.stdout):
    """
    Calculate volume displacement from motion parameter
    """
    
    output = dict()
    columns = volreg.columns
    # Framewise displacement
    io_handler.write('Calculating Displacement from motion parameters...')
    output['FD'] = np.abs(np.insert(np.diff(volreg, axis=0), 0, 0, axis=0)).sum(axis=1)
    # Absolute rotational displacement
    output['ARD'] = np.abs(np.insert(np.diff(volreg[columns[:3]], axis=0), 0, 0, axis=0)).sum(axis=1)
    # Absolute translational displacement
    output['ATD'] = np.abs(np.insert(np.diff(volreg[columns[3:]], axis=0), 0, 0, axis=0)).sum(axis=1)
    io_handler.write('[Done]\n')
    return pd.DataFrame(output)