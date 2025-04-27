import numpy as np
from scipy.linalg import qr as scipy_qr


def solve_normal_equation(X, y):
    """
    Solve OLS via normal equation and also compute effects = Q^T y.

    Returns
    -------
    beta : ndarray, shape (n_features, n_targets)
    effects : ndarray, shape (n_features, n_targets)
    """
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)
    R = np.linalg.cholesky(XtX).T
    effects = R @ beta
    return beta, effects


def solve_qr(X: np.ndarray, y: np.ndarray, tol: float = 1e-7):
    """
    Solve linear system using QR decomposition with column pivoting.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Design matrix.
    y : np.ndarray, shape (n_samples, n_targets)
        Response matrix or vector.
    tol : float
        Tolerance for determining rank deficiency.

    Returns
    -------
    beta : np.ndarray, shape (n_features, n_targets)
        Estimated coefficients, in original column order.
    effects : np.ndarray, shape (n_features, n_targets)
        Raw effects: Q^T @ y for each orthonormal basis vector.
    piv : np.ndarray, shape (n_features,)
        Pivot indices applied to columns.
    """
    # 1) QR decomposition with pivoting
    Q, R, piv = scipy_qr(X, mode='economic', pivoting=True)
    # 2) compute effects = Q^T y
    effects = Q.T @ y
    # 3) determine rank
    diag_R = np.abs(np.diag(R))
    rank = np.sum(diag_R > tol)
    # 4) solve R @ z = effects
    z = np.linalg.solve(R[:rank, :rank], effects[:rank])
    # 5) reconstruct beta in pivoted order
    beta = np.zeros((X.shape[1], y.shape[1] if y.ndim > 1 else 1))
    beta[piv[:rank], :] = z
    return beta, effects


def safe_divide(numerator, denominator):
    """
    Elementwise division, but where denominator==0, yield NaN (not a tiny epsilon).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),    # fill zeros where division is invalid
        where=(denominator != 0)         # only divide where denom ≠ 0
    )
    return result

