import nibabel as nib
import numpy as np

from camri.connectivity import (
    alff_image,
    correlation_matrix,
    kendall_w,
    reho,
    reho_image,
)
from camri.image import apply_mask, crop_image, resample_voxels, smooth_image, unmask
from camri.processing import regress_confounds, standardize, temporal_filter
from camri.qc import dvars, dvars_image, framewise_displacement, tsnr, tsnr_image


def _image(shape=(5, 6, 7, 8)):
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    affine = np.diag([2, 2, 2, 1]).astype(float)
    return nib.Nifti1Image(data, affine)


def test_image_helpers_preserve_world_origin_and_grid():
    image = _image((5, 6, 7, 2))
    cropped = crop_image(image, (1, 2, 3), (4, 6, 7))
    assert cropped.shape == (3, 4, 4, 2)
    assert np.allclose(cropped.affine[:3, 3], [2, 4, 6])
    resampled = resample_voxels(cropped, (1, 1, 1))
    assert np.allclose(np.linalg.norm(resampled.affine[:3, :3], axis=0), [1, 1, 1])


def test_mask_smoothing_and_unmask():
    image = _image((5, 5, 5, 3))
    mask = nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.uint8), image.affine)
    masked = apply_mask(image, mask)
    assert masked.shape == image.shape
    assert np.isfinite(smooth_image(image, 2, mask).get_fdata()).all()
    values = np.arange(int(np.asarray(mask.dataobj).sum()), dtype=float)
    assert unmask(values, mask).shape == mask.shape


def test_signal_processing_has_explicit_constant_and_rank_rules():
    assert np.isnan(standardize(np.ones((2, 4)))[0]).all()
    result = regress_confounds(np.arange(8.0).reshape(2, 4), np.ones((4, 1)))
    assert result.rank == 1
    filtered = temporal_filter(np.sin(np.linspace(0, 20, 100)), 1.0, highcut=0.2)
    assert filtered.shape == (100,)


def test_qc_and_connectivity_metrics():
    signals = np.vstack([np.arange(6.0), np.arange(6.0) * 2 + 1])
    assert np.isfinite(tsnr(signals)).all()
    assert dvars(signals).shape == (5,)
    motion = np.zeros((3, 6))
    motion[1, 0] = 1
    assert np.allclose(framewise_displacement(motion, radius_mm=10), [0, 10, 10])
    corr = correlation_matrix(signals)
    assert np.allclose(corr[0, 1], 1)


def test_reho_ties_are_finite_for_nontrivial_fixture():
    data = np.stack([np.arange(8), np.arange(8), np.arange(8) ** 2], axis=0).astype(float)
    assert np.isfinite(kendall_w(data))
    image = np.zeros((3, 3, 3, 8), dtype=float)
    image[1, 1, 1] = np.arange(8)
    image[1, 1, 2] = np.arange(8)
    result = reho(image, mask=np.ones((3, 3, 3), dtype=bool), neighborhood=6)
    assert np.isfinite(result[1, 1, 1])


def test_image_metric_adapters_return_expected_grids():
    image = _image((3, 3, 3, 32))
    mask = nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.uint8), image.affine)
    assert tsnr_image(image, mask=mask).shape == (3, 3, 3)
    assert dvars_image(image, mask=mask).shape == (31,)
    assert alff_image(image, 1.0, mask=mask, lowcut=0.01, highcut=0.2).shape == (3, 3, 3)
    assert reho_image(image, mask=mask).shape == (3, 3, 3)
