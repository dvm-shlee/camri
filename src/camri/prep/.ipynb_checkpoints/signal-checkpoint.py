import sys
import numpy as np


def get_nonzero_signal(data):
    mask = np.nonzero(data.std(-1) != 0)
    empty_data = np.zeros_like(data)
    return data[mask], mask, empty_data


def signal_normalization(data: np.ndarray, method='standardize', **kwargs):
    from scipy import stats
    if method == "standardize":
        return ((data.T - data.mean(-1)) / data.std(-1)).T
    elif method == "mode":
        kwargs.setdefault('mode', 1000)
        return (data / np.median(data)) * kwargs['mode']
    else:
        raise TypeError


def power_spectral_density(data: np.ndarray,
                           dt,
                           nperseg = None,
                           average: str = 'mean'
                           ):
    """Estimate power spectral density using Welch's method.
    """
    from scipy import signal
    fs = 1.0 / dt
    input_length = data.shape[-1]
    if input_length < 256 and nperseg is None:
        nperseg = input_length

    f, pxx = signal.welch(data, fs=fs, window='hann', nperseg=nperseg,
                          scaling='density', average=average)
    return f, pxx


def bandpass(data: np.ndarray,
             dt: float, order: int = 5,
             lowcut = None,
             highcut = None,
             output: str = 'sos',
             analog = False) -> np.ndarray:
    from scipy import signal
    fs = 1.0 / dt
    nyq = 0.5 * fs
    if lowcut and highcut:
        op = signal.butter(order, [lowcut/nyq, highcut/nyq], 
                           btype='bandpass', output=output, analog=analog)
    else:
        if lowcut:
            op = signal.butter(order, lowcut/nyq, 
                               btype='highpass', output=output, analog=analog)
        elif highcut:
            op = signal.butter(order, highcut/nyq, 
                               btype='lowpass', output=output, analog=analog)
        else:
            raise TypeError
    if output == 'sos':
        return signal.sosfilt(op, data)
    elif output == 'ba':
        return signal.lfilter(op[0], op[1], data)
    else:
        raise TypeError


def polynomial_feature(data: np.ndarray,
                       order: int) -> np.ndarray:
    """ data: V x T data where V is voxels and T is time points
        order: order of polynomial
    Return:
        model: F x T
    """
    n_ft = order + 1
    n_dp = data.shape[-1]
    model = np.zeros([n_ft, n_dp])
    for order in range(n_ft):
        if order == 0:
            model[order, :] = np.ones(n_dp)
        else:
            x = np.arange(n_dp)
            model[order, :] = x ** order
    model = (model.T / model.max(axis=1, initial=None)).T
    return model


def linear_regression(data: np.ndarray,
                      model: np.ndarray,
                      method: str = 'svd'):
    """ linear regression using linear decomposition algorithm
    data: V x T data where V is number of voxels and T is number of time points
    model: F x T data where T is number of time points and F is number of features
   
    beta_coefficient : V x F
    """
    model = model.copy().T
    data = data.copy().T
    if method == 'svd':
        bs = np.linalg.pinv(model).dot(data).T
    elif method == 'qr':
        q, r = np.linalg.qr(model)
        bs = np.linalg.inv(r).dot(q.T).dot(data).T
    else:
        raise TypeError
    return bs


def nuisance_regression(data,
                        port: int =1,
                        ort = None):
    model = polynomial_feature(data, order=port)
    if isinstance(ort, np.ndarray):
        ort = ((ort.T - ort.mean(-1)) / abs(ort).max(-1)).T
        model = np.concatenate([model, ort], axis=0)
    beta_coef = linear_regression(data, model=model)
    filt_data = data - beta_coef @ model
    return filt_data, beta_coef