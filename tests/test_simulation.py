import numpy as np
import pytest
from ramanspec.simulation import SimulationConfig, gaussian, simulate


def test_shapes_truth_reproducibility():
    config = SimulationConfig(rows=6, cols=6, channels=121)
    a, b = simulate(config), simulate(config)
    assert a.observed.shape == (36, 121)
    assert a.coordinates.shape == (36, 2)
    assert set(a.labels) == {0, 1, 2}
    assert (a.axis[0], a.axis[-1]) == (400, 1800)
    assert np.all(np.diff(a.axis) > 0)
    np.testing.assert_array_equal(a.observed, b.observed)
    np.testing.assert_allclose(a.observed, a.clean + a.baseline + a.noise + a.spikes)
    np.testing.assert_array_equal(a.spike_mask, a.spikes > 0)
    assert not np.array_equal(a.observed, simulate(SimulationConfig(rows=6, cols=6, channels=121, seed=43)).observed)


def test_gaussian_peak():
    axis = np.arange(400, 1801)
    peak = gaussian(axis, 2, 1000, 10)
    assert axis[np.argmax(peak)] == 1000
    assert peak.max() == 2
    assert peak[590] == peak[610]


@pytest.mark.parametrize('kwargs', [{'noise': -1}, {'channels': 2}, {'spike_probability': 2}, {'seed': -1}, {'rows': 1.5}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        SimulationConfig(**kwargs)
