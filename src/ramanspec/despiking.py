"""Positive local-median residual detection, scaled MAD, and linear repair."""
import numpy as np
from scipy.ndimage import median_filter
from .preprocessing import validate_spectra


def despike(data, window: int = 7, threshold: float = 8.0, max_width: int = 2):
    """Return corrected copy and mask. Reject broad candidate runs to protect peaks.

    Noise scale uses the larger MAD estimate from median residuals and
    adjacent differences (scaled by sqrt(2)) for each spectrum; a numerical
    floor handles zero-noise spectra. Genuine unresolved peaks remain ambiguous.
    """
    array = validate_spectra(data)
    if not isinstance(window, int) or window < 3 or window % 2 != 1 or window > array.shape[-1]:
        raise ValueError('window must be odd, >= 3, and no longer than the spectrum')
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('threshold must be positive and finite')
    if not isinstance(max_width, int) or not 1 <= max_width < window:
        raise ValueError('max_width must be between 1 and window-1')
    result = array.copy()
    mask = np.zeros(array.shape, dtype=bool)
    for row, output, flags in zip(array.reshape(-1, array.shape[-1]),
                                  result.reshape(-1, array.shape[-1]), mask.reshape(-1, array.shape[-1])):
        residual = row - median_filter(row, size=window, mode='reflect')
        center = np.median(residual)
        differences = np.diff(row)
        difference_scale = 1.4826 * np.median(np.abs(differences - np.median(differences))) / np.sqrt(2)
        scale = max(1.4826 * np.median(np.abs(residual - center)), difference_scale,
                    1e-8 * max(1., np.ptp(row)))
        candidates = np.flatnonzero(residual - center > threshold * scale)
        for run in np.split(candidates, np.flatnonzero(np.diff(candidates) > 1) + 1):
            if 0 < run.size <= max_width:
                flags[run] = True
        valid = ~flags
        if valid.sum() < 2:
            raise ValueError('Too few unflagged channels for interpolation')
        output[flags] = np.interp(np.flatnonzero(flags), np.flatnonzero(valid), row[valid])
    return result, mask
