import matplotlib.pyplot as plt
import numpy as np
from camri.stats.metrics.regression import RegressionMetrics

class Evaluator:
    """
    Evaluator class for regression models.
    Computes standard metrics for multi-target (e.g., multi-voxel) outputs.
    """

    def __init__(self, estimator, y_true):
        self.estimator = estimator
        self.y_true = y_true

        # Check if estimator has fitted predictions
        if hasattr(estimator, 'y_pred_'):
            self.y_pred = estimator.y_pred_
        else:
            raise AttributeError("Estimator must have fitted predictions (y_pred_) after calling fit().")

    def evaluate(self):
        """
        Evaluate the model and return a dictionary of per-voxel metrics.
        """
        n_features = self.estimator.coef_.shape[0]
        return {
            'mse': RegressionMetrics.mse(self.y_true, self.y_pred),
            'rmse': RegressionMetrics.rmse(self.y_true, self.y_pred),
            'r2': RegressionMetrics.r2(self.y_true, self.y_pred),
            'adjusted_r2': RegressionMetrics.adjusted_r2(self.y_true, self.y_pred, n_features)
        }

    def plot_residuals(self, flatten=True, method="hist", bins=50, plot_summary=False):
        """
        Plot residuals after fitting.

        Parameters
        ----------
        flatten : bool
            If True, flatten (n_samples, n_voxels) residuals into 1D for plotting.
        method : str
            "hist" for histogram, "scatter" for residual vs predicted scatter plot.
        bins : int
            Number of bins for histogram.
        plot_summary : bool
            If True and flatten=False, plot sample-wise residual mean with std error bars.
        """
        residuals = self.y_true - self.y_pred

        if flatten:
            residuals = residuals.flatten()
            preds = self.y_pred.flatten()
        else:
            residuals_mean = np.mean(residuals, axis=1)  # (n_samples,)
            residuals_std = np.std(residuals, axis=1)    # (n_samples,)
            preds = np.mean(self.y_pred, axis=1)          # (n_samples,)

        plt.figure(figsize=(6, 4))
        
        if plot_summary and not flatten:
            # Special plot: sample-wise residual mean ± std
            x = np.arange(residuals_mean.shape[0])  # sample indices
            plt.errorbar(x, residuals_mean, yerr=residuals_std, fmt='o', ecolor='gray', capsize=3, markersize=4)
            plt.title("Sample-wise Residual Mean ± STD")
            plt.xlabel("Sample Index")
            plt.ylabel("Residual Mean")
            plt.axhline(0, color='red', linestyle='--')
        else:
            if method == "hist":
                plt.hist(residuals, bins=bins, edgecolor='k')
                plt.title("Residuals Histogram")
                plt.xlabel("Residual")
                plt.ylabel("Frequency")
            elif method == "scatter":
                plt.scatter(preds, residuals, alpha=0.5)
                plt.title("Residuals vs Predicted")
                plt.xlabel("Predicted")
                plt.ylabel("Residual")
                plt.axhline(0, color='red', linestyle='--')
            else:
                raise ValueError("method must be 'hist' or 'scatter'")

        plt.grid(True)
        plt.tight_layout()
        plt.show()


