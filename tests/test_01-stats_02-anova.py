# import pytest
# import numpy as np
# from camri.stats.model.ols import OLS
# from camri.stats.inference.anova import Anova

# @pytest.fixture
# def sample_data_multi_voxel():
#     """
#     Generate sample data for multi-voxel linear regression tests.
#     Model: y = 2*x0 + 1.5*x1 + 3 + noise
#     """
#     np.random.seed(42)
#     n_samples = 100
#     n_voxels = 5
#     x0 = np.linspace(0, 10, n_samples).reshape(-1, 1)
#     x1 = np.random.normal(0, 1, size=(n_samples, 1))
#     X = np.hstack([x0, x1])
#     noise = np.random.normal(0, 0.5, size=(n_samples, n_voxels))
#     y = 2 * x0 + 1.5 * x1 + 3 + noise
#     return X, y

# def check_anova_result_structure(results, X, y):
#     """
#     Common checker for ANOVA test results.
#     """
#     required_keys = ["SS", "DF", "MS", "F", "p", "SS_error", "DF_error"]
#     for key in required_keys:
#         assert key in results, f"Missing key {key} in ANOVA results."

#     n_voxels = y.shape[1]
#     n_features = X.shape[1] + 1  # intercept included

#     assert results["SS"].shape == (n_features-1, n_voxels)
#     assert results["DF"].shape == (n_features-1,)
#     assert results["MS"].shape == (n_features-1, n_voxels)
#     assert results["F"].shape == (n_features-1, n_voxels)
#     assert results["p"].shape == (n_features-1, n_voxels)
#     assert results["SS_error"].shape == (n_voxels,)
#     assert isinstance(results["DF_error"], int)

# def test_anova_type1(sample_data_multi_voxel):
#     """
#     Test Type I ANOVA computation.
#     """
#     X, y = sample_data_multi_voxel
#     model = OLS(fit_intercept=True, method="normal")
#     model.fit(X, y)

#     anova = Anova(model, typ="I")
#     anova.fit()
#     results = anova.summary()

#     check_anova_result_structure(results, X, y)

# def test_anova_type2(sample_data_multi_voxel):
#     """
#     Test Type II ANOVA computation.
#     """
#     X, y = sample_data_multi_voxel
#     model = OLS(fit_intercept=True, method="normal")
#     model.fit(X, y)

#     anova = Anova(model, typ="II")
#     anova.fit()
#     results = anova.summary()

#     check_anova_result_structure(results, X, y)

# def test_anova_type3(sample_data_multi_voxel):
#     """
#     Test Type III ANOVA computation.
#     """
#     X, y = sample_data_multi_voxel
#     model = OLS(fit_intercept=True, method="normal")
#     model.fit(X, y)

#     anova = Anova(model, typ="III")
#     anova.fit()
#     results = anova.summary()

#     check_anova_result_structure(results, X, y)
