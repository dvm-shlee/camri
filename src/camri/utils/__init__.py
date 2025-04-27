from . import nifti
from . import roi
from . import dmat
import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def get_matsize_from_arr(arr):
    return int((np.sqrt(arr.shape[0]*8+1)+1)/2)

def arr2mat(arr):
    size = get_matsize_from_arr(arr)
    mat = np.zeros([size] * 2)
    mat[np.tril_indices(size, k=-1)] = arr
    return mat

def mat2arr(mat):
    return mat[np.tril_indices(mat.shape[0], k=-1)]


__all__ = ['nifti', 'roi', 'dmat', 'bcolors', 'arr2mat', 'mat2arr']