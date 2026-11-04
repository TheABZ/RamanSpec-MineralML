import numpy as np
import pytest
from ramanspec.preprocessing import validate_spectra, vector_normalize, preprocess


def test_normalization_and_zero():
    result = vector_normalize([[3., 4., 0.], [0., 0., 0.]])
    np.testing.assert_allclose(result[0], [.6, .8, 0])
    np.testing.assert_array_equal(result[1], 0)


def test_missing_policy():
    y = [1, np.nan, 3, np.inf, 5]
    with pytest.raises(ValueError):
        validate_spectra(y)
    np.testing.assert_allclose(validate_spectra(y, 'interpolate'), [1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        validate_spectra([np.nan]*3, 'interpolate')


def test_batch_scaling():
    data = np.random.default_rng(0).normal(size=(8, 101))
    result, info = preprocess(data, standardize=True)
    assert result.shape == data.shape
    np.testing.assert_allclose(result.mean(axis=0), 0, atol=1e-12)
    assert info['scaler'] is not None
