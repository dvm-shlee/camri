import numpy as np
import pandas as pd
import pytest

# Import the OLS class from your module; adjust the path as needed
from camri.stats.model import OLS


def generate_linear_data(n=100, noise=0.0, multi=False):
    """
    Create a simple linear dataset: y = 2 + 3*x + noise.
    If multi=True, returns two targets with different slopes.
    """
    np.random.seed(0)
    x = np.linspace(0, 1, n)
    y_base = 2 + 3 * x + noise * np.random.randn(n)
    df = pd.DataFrame({'x': x})
    if not multi:
        y = y_base.reshape(-1, 1)
    else:
        # Second target: y2 = 1 + 4*x
        y2 = 1 + 4 * x + noise * np.random.randn(n)
        y = np.vstack([y_base, y2]).T
    return df, y


def test_simple_linear_fit_predict():
    df, y = generate_linear_data(n=50, noise=0.0)
    model = 'y ~ x'
    ols = OLS(model, df, y)
    fitted = ols.fit()

    # Check intercept and slope
    assert np.allclose(fitted.intercept_, [[2.0]], atol=1e-6)
    assert np.allclose(fitted.coef_, [[3.0]], atol=1e-6)

    # Predictions match true values exactly
    y_pred = fitted.predict()
    assert y_pred.shape == y.shape
    assert np.allclose(y_pred, y, atol=1e-6)

    # Residuals are zero for noiseless data
    resid = fitted.resid
    assert np.allclose(resid, np.zeros_like(resid), atol=1e-8)


def test_multiple_targets():
    df, y = generate_linear_data(n=60, noise=0.0, multi=True)
    model = 'y ~ x'
    ols = OLS(model, df, y)
    fitted = ols.fit()

    # Two intercepts and two slopes
    assert fitted.intercept_.shape == (1, 2)
    assert fitted.coef_.shape == (1, 2)
    assert np.allclose(fitted.intercept_[0, 0], 2.0, atol=1e-6)
    assert np.allclose(fitted.coef_[0, 1], 4.0, atol=1e-6)

    # R^2 should be 1 for both targets without noise
    stats = fitted.statistics_
    assert 'r2' in stats
    assert np.allclose(stats['r2'], np.ones(2), atol=1e-8)


def test_predict_before_fit_raises():
    df, y = generate_linear_data(n=10)
    model = 'y ~ x'
    ols = OLS(model, df, y)
    with pytest.raises(ValueError):
        _ = ols.predict()


def test_resid_property_before_fit_raises():
    df, y = generate_linear_data(n=10)
    model = 'y ~ x'
    ols = OLS(model, df, y)
    with pytest.raises(ValueError):
        _ = ols.resid
