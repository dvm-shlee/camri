import numpy as np
from scipy.stats import t
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
    
    def marginal_summary(self, contrast, alpha=0.05):
        """
        Compute marginal predicted values, standard errors, and confidence intervals
        for a given linear contrast from a fitted OLS model.

        Parameters
        ----------
        contrast : array_like, shape (n_features,)
            Contrast vector R specifying the linear combination of coefficients.
        alpha : float, default=0.05
            Significance level for confidence intervals (1-alpha CI).

        Returns
        -------
        summary : dict
            {
            'pred': ndarray (n_targets,),     # predicted marginal means
            'se':   ndarray (n_targets,),     # standard errors
            'ci_lower': ndarray (n_targets,), # lower bound of CI
            'ci_upper': ndarray (n_targets,), # upper bound of CI
            't_df': int,                      # degrees of freedom
            't_crit': float                   # critical t value for CI
            }

        Notes
        -----
        Prediction: R @ beta
        Var(pred) = sigma^2 * (R @ (X'X)^{-1} @ R.T)
        SE = sqrt(var(pred))
        CI: pred ± t_{df,1-alpha/2} * SE
        """
        # Stack intercept and coefficients
        beta = np.vstack([self.intercept_, self.coef_])  # (n_features, n_targets)
        # Compute predicted marginal means
        pred = contrast @ beta                                # (n_targets,)

        # Degrees of freedom and residual variance per target
        df_resid = self.statistics_["dof"]["error"]                               # (n_samples, n_targets)
        sigma2 = self.statistics_["residual_variance"]       # (n_targets,)

        # Compute covariance factor
        XtX_inv = self.statistics_["inv_XtX"]                    # (n_features, n_features)
        var_factor = contrast @ XtX_inv @ contrast.T          # scalar

        # Standard error of prediction for each target
        se = np.sqrt(sigma2 * var_factor)                     # (n_targets,)

        # Critical t-value
        t_crit = t.ppf(1 - alpha/2, df_resid)

        # Confidence intervals
        ci_lower = pred - t_crit * se
        ci_upper = pred + t_crit * se

        return {
            'pred': pred,
            'se': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            't_df': df_resid,
            't_crit': t_crit
        }
