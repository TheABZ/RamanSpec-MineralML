import json
import numpy as np
from ramanspec.pipeline import run_demo
from ramanspec.simulation import SimulationConfig


def test_end_to_end_exports(tmp_path):
    config = SimulationConfig(rows=5, cols=6, channels=121)
    metrics = run_demo(tmp_path, config=config, experiments=False)
    assert len(list((tmp_path / 'figures').glob('*.png'))) == 13
    assert json.loads((tmp_path / 'metrics/metrics.json').read_text()) == metrics
    data = np.load(tmp_path / 'data/spectra.npz')
    assert data['observed'].shape == (30, 121)
    assert (tmp_path / 'reports/summary.md').is_file()
    assert (tmp_path / 'metrics/predictions.csv').is_file()
