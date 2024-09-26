import sys
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from typing import Optional, Union
from typing import Tuple
from .signal import power_spectral_density


def r_to_t(r: float, size: int) -> Tuple[float, float]:
    """
    Convert a correlation coefficient to a t-value and compute its corresponding p-value.
    
    Parameters:
    - r : float
        The correlation coefficient.
    - size : int
        The sample size.

    Returns:
    - tval : float
        The t-value corresponding to the given correlation.
    - pval : float
        The p-value corresponding to the t-value.
    """
    from scipy.stats import t
    try:
        tval = r * np.sqrt(size - 2) / np.sqrt(1 - np.square(r))
        pval = 1 - t.cdf(tval, size - 2)
    except:
        r = r.astype(np.float32)
        tval = (r * np.sqrt(size - 2) / np.sqrt(1 - np.square(r))).astype(np.float32)
        pval = (1 - t.cdf(tval, size - 2)).astype(np.float32)
    return tval, pval


def corr(x: np.ndarray) -> float:
    """
    Compute the auto-correlation of a given 1D array.
    
    Parameters:
    - x : np.ndarray
        The input 1D array.

    Returns:
    - r : float
        The auto-correlation value.
    """
    vals = np.zeros(x.shape)
    mask = np.nonzero(x.std(-1))

    try:
        vals[mask] = ((x[mask].T - x[mask].mean(-1)) / x[mask].std(-1)).T
    except:
        vals[mask] = ((x[mask].T - x[mask].mean(-1)) / x[mask].std(-1)).T.astype(np.float32)
    r = np.dot(vals, vals.T) / vals.shape[-1]
    return r


def corr_with(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the cross-correlation between two 1D arrays.
    
    Parameters:
    - x, y : np.ndarray
        The input 1D arrays.

    Returns:
    - r : float
        The cross-correlation value.
    """
    val_x = np.zeros(x.shape)
    val_y = np.zeros(y.shape)

    x_mask = np.nonzero(x.std(-1))
    y_mask = np.nonzero(y.std(-1))
    try:
        val_x[x_mask] = ((x[x_mask].T - x[x_mask].mean(-1)) / x[x_mask].std(-1)).T
        val_y[y_mask] = ((y[y_mask].T - y[y_mask].mean(-1)) / y[y_mask].std(-1)).T
    except:
        val_x[x_mask] = ((x[x_mask].T - x[x_mask].mean(-1)) / x[x_mask].std(-1)).T.astype(np.float32)
        val_y[y_mask] = ((y[y_mask].T - y[y_mask].mean(-1)) / y[y_mask].std(-1)).T.astype(np.float32)
    r = np.dot(val_x, val_y.T) / x.shape[-1]
    return r


def phase_locking_value(x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute the phase-locking value (PLV) between two 1D arrays.
    
    Parameters:
    - x, y : np.ndarray
        The input 1D arrays representing signal amplitudes.

    Returns:
    - float
        The PLV value between x and y.
    - np.ndarray
        Difference in phase angles between x and y.
    """
    x_phase = np.angle(hilbert(x), deg=False)
    y_phase = np.angle(hilbert(y), deg=False)
    angle_diff = x_phase - y_phase
    return abs(np.exp(1j*angle_diff).mean()), angle_diff


def const_maxcorr(df: pd.DataFrame, dt: float, max_lag: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute the maximum cross-correlation within a constrained time lag for time series data in a DataFrame.
    
    Parameters:
    - df : pd.DataFrame
        Time series data where rows are timepoints and columns are different regions of interest (ROIs).
    - dt : float
        Sampling interval in seconds.
    - max_lag : float
        The constrained maximum lag in seconds.

    Returns:
    - pd.DataFrame
        Matrix of maximum correlations for each ROI pair.
    - pd.DataFrame
        Matrix of lags (in timepoints) at which the maximum correlations occur.
    """
    
    max_lag = int(max_lag / dt)
    all_lags = np.arange(-max_lag, max_lag + 1)
    max_corr = pd.DataFrame(np.zeros([df.shape[-1]] * 2), index=df.columns,
                            columns=df.columns)
    max_corr_lag = max_corr.copy()

    for col_id1, ts1 in df.iteritems():
        cross_corr = np.zeros([len(all_lags), len(df.columns)])
        for col_id2, ts2 in df.iteritems():
            for lag_id, lag in enumerate(all_lags):
                cross_corr[lag_id, col_id2] = ts1.corr(ts2.shift(lag))
        max_lag_idxs = abs(cross_corr).argmax(0)
        for col_id2, max_lag_idx in enumerate(max_lag_idxs):
            max_corr.loc[col_id1, col_id2] = cross_corr[max_lag_idx, col_id2]
            max_corr_lag.loc[col_id1, col_id2] = all_lags[max_lag_idx]
    return max_corr, max_corr_lag


def connectivity_strength(x: np.ndarray, y: Optional[np.ndarray] = None,
                          pval: Optional[float] = None,
                          pos=False, abs=False) -> np.ndarray:
    if y is None:
        r = corr(x)
        r[np.nonzero(np.eye(r.shape[0]))] = 0
    else:
        r = corr_with(x, y)

    if pos:
        r[r < 0] = 0
    if abs:
        r = np.abs(r)
    if pval is not None:
        t, p = r_to_t(r, x.shape[-1])
        r[p >= pval] = 0
    r[np.nonzero(np.eye(r.shape[0]))] = 0
    return r.sum(-1)

def cal_distance(coord_a: Tuple[float, float, float], coord_b: Tuple[float, float, float]) -> float:
    """
    Calculate the Euclidean distance between two 3D coordinates.
    
    Parameters:
    - coord_a, coord_b : Tuple[float, float, float]
        The two 3D coordinates.

    Returns:
    - float
        The Euclidean distance between the two coordinates.
    """
    return np.sqrt(np.square(np.diff(np.asarray(list(zip(coord_a, coord_b))))).sum())


def get_cluster_coordinates(coord: Tuple[int, int, int], size: int=1, nn_level: int=3) -> Tuple[int, int, int]:
    """
    Generate a list of 3D coordinates representing a cluster around the provided center.
    
    Parameters:
    - coord : Tuple[int, int, int]
        The center of the cluster.
    - size : int, optional (default=1)
        The size of the cluster from the center.
    - nn_level : int, optional (default=3)
        Defines the type of neighbors to include: 
        1 = faces only, 
        2 = faces and edges, 
        3 = faces, edges, and corners.

    Returns:
    - list of tuples
        List of 3D coordinates representing the cluster.
    """
    n_voxel = size + 1
    x, y, z = coord
    x_ = sorted([x + i for i in range(n_voxel)] + [x - i for i in range(n_voxel) if i != 0])
    y_ = sorted([y + i for i in range(n_voxel)] + [y - i for i in range(n_voxel) if i != 0])
    z_ = sorted([z + i for i in range(n_voxel)] + [z - i for i in range(n_voxel) if i != 0])

    if nn_level == 1:
        thr = size
    elif nn_level == 2:
        thr = np.sqrt(np.square([size] * 2).sum())
    elif nn_level == 3:
        thr = np.sqrt(np.square([size] * 3).sum())
    else:
        raise ValueError('[nn_level] only accept a value in [1, 2, 3]')

    all_poss = [(i, j, k) for i in x_ for j in y_ for k in z_]
    output_coord = [c for c in all_poss if cal_distance(coord, c) <= thr]

    return output_coord


def kendall_w(data: np.ndarray) -> float:
    """
    Compute Kendall's coefficient of concordance (W) for the provided data.
    
    The coefficient W is used to assess the agreement among multiple rankers/observers.

    Parameters:
    -----------
    data : np.ndarray
        A 2D numpy array where rows represent rankers or observers and columns represent 
        the items being ranked. Each cell contains the rank given by a particular ranker
        to a particular item.

    Returns:
    --------
    float
        The computed Kendall's W value. A value closer to 1 indicates stronger agreement
        among rankers, while a value closer to 0 indicates weaker agreement.

    Example:
    --------
    >>> data = np.array([[1, 2, 3], [1, 3, 2], [2, 1, 3]])
    >>> kendall_w(data)
    0.5

    """
    m, n = data.shape

    if m != 0:
        # Compute ranks for each ranker/observer
        temp = data.argsort(axis=1)
        ranks = temp.argsort(axis=1).astype(np.float64) + 1
        # Sum ranks for each item
        ranks_sum = ranks.sum(axis=0)
        # Compute mean rank for each item
        mean_ranks = ranks_sum.mean()
        # Calculate the sum of squared deviations from the mean ranks
        ssd = np.square(ranks_sum - mean_ranks).sum()
        # Calculate Kendall's W
        w = 12 * ssd / (m ** 2 * (n ** 3 - n))
        return w
    else:
        return 0
    

def regional_homogeneity(data, nn=3, io_handler=sys.stdout):
    from functools import partial
    mask_idx = np.nonzero(data.mean(-1))
    gcc = partial(get_cluster_coordinates, size=1, nn_level=nn)
    io_handler.write('Extracting nearest coordinates set...')
    all_coords_set = np.apply_along_axis(gcc, 0, np.array(mask_idx)).T
    io_handler.write('[Done]\n')
    masked_reho = np.zeros(all_coords_set.shape[0])

    n_voxels = all_coords_set.shape[0]
    progress = 1
    io_handler.write('Calculating regional homogeneity...\n')
    for i, coord in enumerate(all_coords_set):
        # filter outbound coordinate
        c_msk = []
        for j, arr in enumerate(coord):
            s = data.shape[j]
            c_msk.append(np.nonzero(arr > s - 1)[0])
        coord_flt = [f for f in range(coord.shape[-1]) if f not in list(set(np.concatenate(c_msk, 0)))]
        coord = coord[:, coord_flt]
        cd = data[tuple(coord)]
        masked_cd = cd[np.nonzero(cd.mean(-1))]
        masked_reho[i] = kendall_w(masked_cd)
        if (i / n_voxels) * 10 >= progress:
            io_handler.write(f'{progress}..')
            progress += 1
        if i == (n_voxels - 1):
            io_handler.write('10 [Done]\n')

    reho = np.zeros(data.shape[:3])
    reho[mask_idx] = masked_reho
    return reho


def amplitude_low_freq_fluctuation(data: np.ndarray,
                                   dt: Union[int, float],
                                   lowcut: Optional[float] = None, highcut: Optional[float] = None,
                                   pval: Optional[float]=None,
                                   fraction: bool=False,
                                   io_handler=sys.stdout):
    """ Amplitude of Low Frequency Fluctuation

    Args:
        data: V x T
        dt: sampling time
        lowcut: cut frequency for highpass filter
        highcut: cut frequency for lowpass filter

    Returns:
        ALFF
    """
    io_handler.write('Calculating ALFF...')
    f, pxx = power_spectral_density(data, dt=dt)
    alff = pxx[..., (f >= lowcut) & (f <= highcut)].sum(-1)
    if fraction:
        alff[np.nonzero(pxx.sum(-1))] /= pxx.sum(-1)[np.nonzero(pxx.sum(-1))]
    io_handler.write('[Done]\n')
    return alff