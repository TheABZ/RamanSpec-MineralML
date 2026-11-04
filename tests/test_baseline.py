import numpy as np
import pytest
from ramanspec.baseline import asls
from ramanspec.simulation import gaussian
from ramanspec.validation import rmse


def test_linear_baseline_and_peaks():
    x = np.linspace(0, 1, 601)
    baseline = .2 + .4*x
    observed = baseline + gaussian(x, 1, .35, .015) + gaussian(x, .5, .7, .02)
    estimate, info = asls(observed, return_info=True)
    assert rmse(baseline, estimate) < .02
    assert info['converged']
    np.testing.assert_allclose(asls(baseline), baseline, atol=1e-6)


@pytest.mark.parametrize('kwargs', [{'lam': 0}, {'p': 0}, {'p': 1}, {'max_iter': 0}, {'tol': -1}, {'lam': np.inf}])
def test_parameter_validation(kwargs):
    with pytest.raises(ValueError):
        asls(np.arange(20.), **kwargs)


def test_iteration_limit_reports_nonconvergence():
    _, info = asls(np.sin(np.arange(30.)), max_iter=1, return_info=True)
    assert not info['converged']
