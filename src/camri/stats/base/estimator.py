import numpy as np

class BaseEstimator:
    """
    Minimal Base Estimator class (vectorized version).
    All estimators should inherit from this class.
    Enforces numpy array inputs and efficient batch operations.
    """
    def __init__(self, **kwargs):
        """
        Initialize estimator with hyperparameters.
        No computation should be done here.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def fit(self, X, y=None):
        """
        Override this method to implement model fitting.
        """
        raise NotImplementedError("The fit() method must be implemented by subclass.")

    def get_params(self, deep=True):
        """
        Return all hyperparameters as a dictionary.
        """
        return {key: getattr(self, key) for key in self.__dict__}

    def set_params(self, **params):
        """
        Set hyperparameters from a dictionary.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _check_X(self, X):
        """
        Validate X is a numpy array, enforce float64 type.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float64)
        else:
            X = X.astype(np.float64, copy=False)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if not np.isfinite(X).all():
            raise ValueError("Input X contains NaN or Inf.")

        return X

    def _check_Xy(self, X, y):
        """
        Validate X and y are numpy arrays, enforce float64 type.
        """
        X = self._check_X(X)
        if y is not None:
            if not isinstance(y, np.ndarray):
                y = np.array(y, dtype=np.float64)
            else:
                y = y.astype(np.float64, copy=False)

            if y.ndim == 1:
                y = y.reshape(-1, 1)

            if not np.isfinite(y).all():
                raise ValueError("Input y contains NaN or Inf.")

            if X.shape[0] != y.shape[0]:
                raise ValueError("X and y must have the same number of samples.")

        return X, y