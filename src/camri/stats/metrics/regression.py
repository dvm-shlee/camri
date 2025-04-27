import numpy as np
from scipy.stats import t
from scipy.linalg import qr as scipy_qr
from ...utils.dmat import strip_termname
from patsy.design_info import DesignMatrix

class RegressionMetrics:
    """
    Standard regression metrics and statistics for model evaluation.
    
    - mse: model evaluation metric (mean squared error)
    - residual_variance: regression analysis metric (s² for standard error, t-tests, CI)
    """

    @staticmethod
    def mse(y_true, y_pred):
        """
        Mean Squared Error (MSE) per voxel for model evaluation
        """
        residuals = y_true - y_pred
        return np.mean(residuals ** 2, axis=0)

    @staticmethod
    def rmse(y_true, y_pred):
        """
        Root Mean Squared Error (RMSE) per voxel
        """
        return np.sqrt(RegressionMetrics.mse(y_true, y_pred))

    @staticmethod
    def r2(y_true, y_pred):
        """
        Coefficient of Determination (R²) per voxel
        """
        ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
        return 1 - ss_res / ss_tot

    @staticmethod
    def adjusted_r2(y_true, y_pred, n_features):
        """
        Adjusted R² per voxel
        """
        n_samples = y_true.shape[0]
        r2 = RegressionMetrics.r2(y_true, y_pred)
        return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

    @staticmethod
    def _compute_residual_variance(y_true, y_pred, df_err):
        """
        Compute residual variance (s²) for regression analysis.

        Returns
        -------
        s2 : ndarray
            Residual variance per voxel
        ss_err : ndarray
            Sum of squared residuals per voxel
        """
        residuals = y_true - y_pred
        ss_err = np.sum(residuals ** 2, axis=0)
        s2 = ss_err / df_err
        return s2, ss_err
    
    @staticmethod
    def _compute_column_dof(dmat, tol: float = 1e-7):
        """
        Compute degrees of freedom for each model term using pivoted QR.

        This function:
        1. Converts the design matrix to a NumPy array and performs a
            pivoted QR decomposition to identify independent columns.
        2. Marks each independent column with df=1 and dependent columns with df=0.
        3. Aggregates column-level degrees of freedom for each term based on
            dmat.design_info.term_name_slices.

        Parameters
        ----------
        dmat : Patsy DesignMatrix or pandas DataFrame with design_info
            The design matrix. Must have `design_info.term_name_slices` mapping
            term names to column indices or slices.
        tol : float, optional
            Tolerance for determining rank deficiency from R diagonal. Default is 1e-7.

        Returns
        -------
        dof : dict
            A dictionary mapping each term name to its computed degrees of freedom.
        """
        # Convert to array
        X = np.asarray(dmat)

        # Perform pivoted QR decomposition
        _, R, piv = scipy_qr(X, mode='economic', pivoting=True)
        diagR = np.abs(np.diag(R))
        rank = np.sum(diagR > tol)

        # Assign df per column: 1 for independent, 0 otherwise
        df_col = np.zeros(X.shape[1], dtype=int)
        df_col[piv[:rank]] = 1

        # Sum df for each term
        dof = []
        for term, slc in dmat.design_info.term_name_slices.items():
            if isinstance(slc, slice):
                idx = np.arange(slc.start, slc.stop)
            else:
                idx = np.asarray(slc)
            dof_name = strip_termname(term)
            dof.append(int(df_col[idx].sum()))
        return dof

    @staticmethod
    def _compute_column_ss(X, effects):
        """
        Compute sum of squares for each model term.

        Parameters
        ---------
        """
        col_names = X.design_info.column_names
        trm_names = X.design_info.term_names
        col_table = np.zeros([len(trm_names), len(col_names)])
        for i, sl in enumerate(X.design_info.term_slices.values()):
            col_table[i, sl] = 1
        ss_fct = col_table @ effects**2
        return ss_fct

    @staticmethod
    def _compute_standard_error(XtX_inv, residual_variance):
        """
        Compute standard error of coefficients per voxel.
        Correctly uses only diagonal elements of covariance matrix.
        """
        diag_cov = np.diag(XtX_inv)  # (n_features,)
        se = np.sqrt(diag_cov[:, None] * residual_variance[None, :])  # (n_features, n_voxels)
        return se

    @staticmethod
    def _compute_tvals_pvals(beta, se, df_err):
        """
        Compute t-values and p-values for coefficients
        """
        tvals = beta / se
        pvals = 2 * t.sf(np.abs(tvals), df_err)
        return tvals, pvals

    @staticmethod
    def _compute_confidence_interval(beta, se, df_err, alpha=0.05):
        """
        Compute confidence intervals for coefficients
        """
        crit_t = t.ppf(1 - alpha / 2, df_err)
        ci_lower = beta - crit_t * se
        ci_upper = beta + crit_t * se
        return ci_lower, ci_upper

    @staticmethod
    def statistics(X, y_true, y_pred, beta, effects, alpha=0.05):
        """
        Compute comprehensive OLS regression statistics.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Design matrix used for fitting.
        y_true : ndarray, shape (n_samples, n_targets)
            Observed responses.
        y_pred : ndarray, shape (n_samples, n_targets)
            Predicted responses from the model.
        beta : ndarray, shape (n_features, n_targets)
            Estimated coefficients.
        effects : ndarray, shape (n_features, n_targets)
            Raw effects (i.e., Q.T @ y) from QR decomposition.
        alpha : float, optional
            Significance level for confidence intervals.

        Returns
        -------
        stats : dict
            A dictionary containing:
            
            - **mse** : ndarray, shape (n_targets,)
            Mean squared error per target.
            - **residual_variance** : ndarray, shape (n_targets,)
            Residual variance (σ²) per target.
            - **se** : ndarray, shape (n_features, n_targets)
            Standard errors for each coefficient.
            - **tvals** : ndarray, shape (n_features, n_targets)
            t-statistics for each coefficient.
            - **pvals** : ndarray, shape (n_features, n_targets)
            Two-sided p-values for each t-statistic.
            - **ci_lower**, **ci_upper** : ndarray, shape (n_features, n_targets)
            Lower and upper bounds of the (1−α) confidence intervals.
            - **r2** : ndarray, shape (n_targets,)
            Coefficient of determination per target.
            - **r2_adj** : ndarray, shape (n_targets,)
            Adjusted R² per target.
            - **ss** : dict
                - **columns** : ndarray, shape (n_features, n_targets)
                Sum of squares for each column (i.e. term).
                - **error** : ndarray, shape (n_targets,)
                Sum of squared residuals.
                - **total** : ndarray, shape (n_targets,)
                Total sum of squares.
            - **dof** : dict
                - **columns** : ndarray, shape (n_features,)
                Degrees of freedom for each column.
                - **error** : int
                Degrees of freedom for error (n_samples − rank(X)).
        """
        n_samples, _ = X.shape
        df_err = X.shape[0] - np.linalg.matrix_rank(X)

        # Residual variance and SSE
        residual_variance, ss_err = RegressionMetrics._compute_residual_variance(
            y_true, y_pred, df_err
        )
        
        # Standard errors, t-values, p-values, and CIs
        XtX = X.T @ X
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(XtX)
        se = RegressionMetrics._compute_standard_error(XtX_inv, residual_variance)
        tvals, pvals = RegressionMetrics._compute_tvals_pvals(beta, se, df_err)
        ci_lower, ci_upper = RegressionMetrics._compute_confidence_interval(
            beta, se, df_err, alpha
        )
        
        # R² and adjusted R²
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
        r2 = 1 - ss_err / ss_tot
        r2_adj = 1 - (1 - r2) * (n_samples - 1) / df_err

        # Sum of squares and degrees of freedom by column
        if isinstance(X, DesignMatrix):
            ss = {
                "column": RegressionMetrics._compute_column_ss(X, effects),
                "error":   ss_err,
                "total":   ss_tot
            }
            dof = {
                "column": RegressionMetrics._compute_column_dof(X),
                "error":   df_err
            }
        else:
            ss, dof = None, None

        return {
            "mse":               RegressionMetrics.mse(y_true, y_pred),
            "residual_variance": residual_variance,
            "se":                se,
            "tvals":             tvals,
            "pvals":             pvals,
            "ci_lower":          ci_lower,
            "ci_upper":          ci_upper,
            "r2":                r2,
            "r2_adj":            r2_adj,
            "ss":                ss,
            "dof":               dof
        }

