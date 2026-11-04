"""Sparse asymmetric least squares with a second-difference roughness penalty."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from .preprocessing import validate_spectra


def asls(spectrum, lam: float = 1e5, p: float = .01, max_iter: int = 30,
         tol: float = 1e-5, return_info: bool = False):
    """Solve (W + lam D.T D) z = W y iteratively; equal residuals get weight 1-p.

    `lam` is for channel-index spacing, so resampling the axis changes its meaning.
    Convergence requires both a small baseline change and stable weights.
    """
    y = validate_spectra(spectrum)
    if y.ndim != 1:
        raise ValueError('asls expects one spectrum')
    if not np.isfinite([lam, p, tol]).all() or lam <= 0 or not 0 < p < 1 or tol <= 0:
        raise ValueError('Require lam > 0, 0 < p < 1 and tol > 0')
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValueError('max_iter must be a positive integer')
    n = y.size
    d = sparse.diags([np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)], [0, 1, 2], shape=(n-2, n), format='csc')
    penalty = lam * (d.T @ d)
    weights = np.ones(n)
    previous = y.copy()
    converged = False
    for iteration in range(1, max_iter + 1):
        z = spsolve(sparse.diags(weights, format='csc') + penalty, weights * y)
        new_weights = np.where(y > z, p, 1-p)
        change = np.linalg.norm(z - previous) / max(np.linalg.norm(previous), 1e-12)
        if change < tol and np.array_equal(weights, new_weights):
            converged = True
            break
        weights, previous = new_weights, z
    info = {'iterations': iteration, 'converged': converged, 'relative_change': float(change)}
    return (z, info) if return_info else z


def estimate_baselines(data, **kwargs) -> np.ndarray:
    """Apply AsLS independently to each row without modifying input."""
    array = validate_spectra(data)
    if kwargs.get('return_info'):
        raise ValueError('Use asls directly to request convergence information')
    if array.ndim == 1:
        return asls(array, **kwargs)
    return np.stack([asls(row, **kwargs) for row in array])
