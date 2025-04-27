import numpy as np
from patsy import dmatrix
from ..base import BaseEstimator
from ..algebra import solve_qr
from ..metrics.regression import RegressionMetrics

class OLS(BaseEstimator):
    """
    Ordinary Least Squares (OLS) Estimator.
    Fully vectorized implementation.
    Supports 'normal' or 'qr' solving methods.
    Handles single or multiple target variables (e.g., multi-voxel).
    """
    def __init__(self, model, df, y):
        super().__init__()
        # terms
        self.terms_ = None
        self.column_names = None
        self.term_names = None
        
        # regression coefficients
        self.coef_ = None
        self.coef_effects_ = None
        self.intercept_ = None
        self.intercept_effect_ = None
        
        # data
        self._encode(model, df, y)
        self.y_pred_ = None
        
        # statistics
        self.statistics_ = None

    def _encode(self, model, df, y):
        lhs, rhs = [e.strip() for e in model.split('~')]
        X = dmatrix(rhs, data=df)
        self.formula = model
        self.df = df
        self.term_names = tuple(X.design_info.term_names)
        self.column_names = tuple(X.design_info.column_names)
        self.X_ = X
        self.y_ = y
        self.y_label_ = lhs

    def fit(self):
        """
        Fit the OLS model.
        """
        X, y = self._check_Xy(self.X_, self.y_)
        self.y_ = y
        # solve for beta
        beta, effects = solve_qr(X, y)

        # split intercept / coef
        self.intercept_ = beta[0:1, :]
        self.intercept_effect_ = effects[0:1, :]
        self.coef_ = beta[1:, :]
        self.coef_effect_ = effects[1:, :]

        # fitted values & residual
        self.y_pred_ = X @ beta

        # compute residuals
        self.resid_ = self.y_ - self.y_pred_
        self.statistics_ = RegressionMetrics.statistics(X, y, self.y_pred_, beta, effects)
        return self

    def predict(self):
        """
        Predict using the OLS model.

        If X is None, return fitted predictions.
        If X is given, predict for new data.
        """
        if self.y_pred_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.y_pred_

    @property
    def resid(self):
        """
        Return residuals.
        """
        if self.y_pred_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.resid_

