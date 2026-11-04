import numpy as np
import pytest
from ramanspec.despiking import despike
from ramanspec.simulation import gaussian
from ramanspec.validation import spike_metrics


def test_detect_repair_preserve_input():
    rng = np.random.default_rng(42)
    clean = gaussian(np.arange(601), 1, 300, 8)
    y = clean + rng.normal(0, .005, 601)
    truth = np.zeros(601, bool)
    truth[[80, 180, 450]] = True
    y[truth] += 2
    original = y.copy()
    repaired, flags = despike(y)
    metrics = spike_metrics(truth, flags)
    assert metrics['recall'] == 1
    assert metrics['precision'] > .7
    assert abs(repaired[300] - y[300]) < .01
    assert np.max(np.abs(repaired[truth] - clean[truth])) < .03
    np.testing.assert_array_equal(original, y)


def test_resolved_narrow_peak_preserved():
    y = gaussian(np.arange(101), 1, 50, 3)
    repaired, _ = despike(y)
    assert repaired.max() > .99


def test_unresolved_peak_is_ambiguous():
    y = np.zeros(101)
    y[50] = 1
    _, flags = despike(y)
    assert flags[50]  # Documented limitation, not experimental peak validation.


@pytest.mark.parametrize('kwargs', [{'window': 4}, {'threshold': 0}, {'max_width': 7}])
def test_invalid(kwargs):
    with pytest.raises(ValueError):
        despike(np.zeros(101), **kwargs)
