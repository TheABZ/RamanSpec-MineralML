"""Reproducible unsupervised models and label-free model selection diagnostics."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def cluster(scores, n_clusters: int = 3, seed: int = 42):
    """Fit both models. Labels, coordinates and clean spectra are never inputs."""
    x = np.asarray(scores, dtype=float)
    if x.ndim != 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError('scores must be a finite nonempty matrix')
    if not isinstance(n_clusters, int) or not 1 <= n_clusters <= x.shape[0]:
        raise ValueError('Invalid cluster count')
    if np.unique(x, axis=0).shape[0] < n_clusters or x.shape[0] < 2:
        raise ValueError('Insufficient distinct observations')
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(x)
    gmm = GaussianMixture(n_components=n_clusters, n_init=3, reg_covar=1e-6, random_state=seed).fit(x)
    if not gmm.converged_:
        raise RuntimeError('GMM did not converge; inspect data or model configuration')
    return {'kmeans': km.labels_, 'gmm': gmm.predict(x), 'probabilities': gmm.predict_proba(x),
            'kmeans_model': km, 'gmm_model': gmm}


def model_selection(scores, counts=(2, 3, 4, 5), seed=42):
    """Smaller AIC/BIC is preferred; inertia always decreases as K grows."""
    rows = []
    for k in counts:
        result = cluster(scores, k, seed)
        rows.append({'clusters': k, 'inertia': result['kmeans_model'].inertia_,
                     'bic': result['gmm_model'].bic(scores), 'aic': result['gmm_model'].aic(scores)})
    return pd.DataFrame(rows)
