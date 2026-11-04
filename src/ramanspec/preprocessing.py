"""Composable preprocessing with an explicit reject-or-interpolate finite policy."""
import numpy as np
from sklearn.preprocessing import StandardScaler


def validate_spectra(data, nonfinite: str = 'raise') -> np.ndarray:
    """Return a copy of a vector or matrix; interpolate missing channels by index."""
    if nonfinite not in ('raise', 'interpolate'):
        raise ValueError('nonfinite must be raise or interpolate')
    array = np.array(data, dtype=float, copy=True)
    if array.ndim not in (1, 2) or array.shape[-1] < 3 or array.size == 0:
        raise ValueError('Expected nonempty spectra with at least three channels')
    for row in array.reshape(-1, array.shape[-1]):
        valid = np.isfinite(row)
        if valid.all():
            continue
        if nonfinite == 'raise' or valid.sum() < 2:
            raise ValueError('Nonfinite data: interpolation requires two finite samples')
        row[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), row[valid])
    return array


def vector_normalize(data) -> np.ndarray:
    """Normalize each spectrum to unit L2 norm; zero spectra remain zero."""
    array = validate_spectra(data)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return np.divide(array, norms, out=np.zeros_like(array), where=norms > 0)


def preprocess(data, *, normalize=True, standardize=False, nonfinite='raise',
               despike_options=None, baseline_options=None):
    """Return processed spectra and diagnostics, using observed spectra only.

    Standardization fits across this batch. For future held-out inference reuse
    the returned scaler rather than fitting on the held-out batch.
    """
    from .despiking import despike
    from .baseline import estimate_baselines
    array = validate_spectra(data, nonfinite)
    corrected, mask = despike(array, **(despike_options or {}))
    baselines = estimate_baselines(corrected, **(baseline_options or {}))
    result = corrected - baselines
    if normalize:
        result = vector_normalize(result)
    scaler = None
    if standardize:
        if result.ndim != 2:
            raise ValueError('Standardization requires a spectral matrix')
        scaler = StandardScaler()
        result = scaler.fit_transform(result)
    return result, {'despiked': corrected, 'spike_mask': mask, 'baseline': baselines, 'scaler': scaler}
