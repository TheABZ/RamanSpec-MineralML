"""Gaussian spectra and separately retained synthetic measurement components."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    """Spatial size determines sample count; noise is in arbitrary intensity units."""
    rows: int = 30
    cols: int = 30
    channels: int = 601
    seed: int = 42
    noise: float = 0.025
    spike_probability: float = 0.002
    shift_min: float = 400.0
    shift_max: float = 1800.0

    def __post_init__(self):
        for name in ('rows', 'cols', 'channels', 'seed'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < (0 if name == 'seed' else 1):
                raise ValueError(f'{name} must be a valid integer')
        if self.channels < 5 or self.rows * self.cols < 3:
            raise ValueError('Need at least 5 channels and 3 spectra')
        if not np.isfinite([self.noise, self.spike_probability, self.shift_min, self.shift_max]).all():
            raise ValueError('Parameters must be finite')
        if self.noise < 0 or not 0 <= self.spike_probability <= 1 or self.shift_min >= self.shift_max:
            raise ValueError('Invalid noise, spike probability or axis range')


@dataclass
class SyntheticDataset:
    axis: np.ndarray
    clean: np.ndarray
    baseline: np.ndarray
    noise: np.ndarray
    spikes: np.ndarray
    observed: np.ndarray
    spike_mask: np.ndarray
    labels: np.ndarray
    coordinates: np.ndarray
    config: SimulationConfig

    def save(self, directory: str | Path) -> None:
        """Save lossless ground truth and human-readable configuration separately."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / 'spectra.npz', **{
            name: value for name, value in vars(self).items() if name != 'config'})
        (directory / 'simulation.json').write_text(json.dumps(asdict(self.config), indent=2), encoding='utf-8')


def gaussian(axis, amplitude: float, center: float, width: float) -> np.ndarray:
    """Gaussian peak; width is standard deviation (FWHM = 2.355 * width)."""
    axis = np.asarray(axis, dtype=float)
    if axis.ndim != 1 or not np.isfinite(axis).all():
        raise ValueError('axis must be a finite vector')
    if not np.isfinite([amplitude, center, width]).all() or width <= 0 or amplitude < 0:
        raise ValueError('Require finite amplitude >= 0 and width > 0')
    return amplitude * np.exp(-0.5 * ((axis - center) / width) ** 2)


def simulate(config: SimulationConfig | None = None) -> SyntheticDataset:
    """Create three invented classes; no literature mineral peak assignment is used."""
    config = config or SimulationConfig()
    rng = np.random.default_rng(config.seed)
    axis = np.linspace(config.shift_min, config.shift_max, config.channels)
    rr, cc = np.indices((config.rows, config.cols))
    coordinates = np.column_stack((rr.ravel(), cc.ravel()))
    # Wavy vertical regions provide contiguous, nontrivial map boundaries.
    position = (cc + 0.5) / config.cols + 0.10 * np.sin(2 * np.pi * rr / config.rows)
    labels = np.clip((3 * position).astype(int), 0, 2).ravel()
    n = labels.size
    clean = np.zeros((n, config.channels))
    signatures = [((600, 1.0, 12), (1000, .70, 18), (1450, .45, 24)),
                  ((650, .85, 15), (1005, .65, 20), (1350, .75, 18)),
                  ((610, .55, 14), (1120, .95, 17), (1500, .80, 22))]
    for i, label in enumerate(labels):
        for center, amplitude, width in signatures[label]:
            clean[i] += gaussian(axis, amplitude * rng.uniform(.85, 1.15),
                                 center + rng.normal(0, 2), width * rng.uniform(.90, 1.10))
    t = (axis - axis.min()) / np.ptp(axis)
    baseline = (rng.uniform(.1, .3, (n, 1)) + rng.uniform(.05, .30, (n, 1)) * t
                + rng.uniform(.1, .4, (n, 1)) * t**2
                + rng.uniform(.1, .4, (n, 1)) * np.exp(-2 * t))
    noise = rng.normal(size=clean.shape) * config.noise * rng.uniform(.8, 1.2, (n, 1))
    spike_mask = rng.random(clean.shape) < config.spike_probability
    spikes = spike_mask * rng.uniform(1.0, 2.5, clean.shape)
    return SyntheticDataset(axis, clean, baseline, noise, spikes,
                            clean + baseline + noise + spikes, spike_mask, labels, coordinates, config)
