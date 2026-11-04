"""PCA scores describe samples; component rows describe spectral directions."""
import numpy as np
from sklearn.decomposition import PCA
from .preprocessing import validate_spectra


def run_pca(data, n_components: int = 5, seed: int = 42):
    """Return scores and fitted PCA; randomized SVD uses an explicit seed."""
    array = validate_spectra(data)
    if array.ndim != 2 or array.shape[0] < 2:
        raise ValueError('PCA needs a matrix of at least two spectra')
    if not isinstance(n_components, int) or not 1 <= n_components <= min(array.shape):
        raise ValueError('n_components outside matrix dimensions')
    if not np.any(np.var(array, axis=0) > 0):
        raise ValueError('PCA requires nonzero variance')
    model = PCA(n_components=n_components, svd_solver='randomized', random_state=seed)
    return model.fit_transform(array), model
