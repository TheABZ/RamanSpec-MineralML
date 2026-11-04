"""Ground truth enters only these evaluation functions and diagnostic plots."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, silhouette_score


def rmse(expected, actual) -> float:
    a, b = np.asarray(expected), np.asarray(actual)
    if a.shape != b.shape or a.size == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('RMSE requires matching finite nonempty arrays')
    return float(np.sqrt(np.mean((a-b)**2)))


def spike_metrics(truth, predicted):
    a, b = np.asarray(truth, bool), np.asarray(predicted, bool)
    if a.shape != b.shape or not a.size:
        raise ValueError('Masks must have matching nonempty shapes')
    tp, fp, fn, tn = [int(v.sum()) for v in (a & b, ~a & b, a & ~b, ~a & ~b)]
    return {'precision': tp/(tp+fp) if tp+fp else 0., 'recall': tp/(tp+fn) if tp+fn else 0.,
            'false_positive_rate': fp/(fp+tn) if fp+tn else 0., 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def evaluate_clusters(truth, predicted, scores, seed=42):
    labels = np.asarray(predicted)
    count = np.unique(labels).size
    silhouette = None
    if 1 < count < len(labels):
        silhouette = float(silhouette_score(scores, labels, sample_size=min(500, len(labels)), random_state=seed))
    return {'adjusted_rand_index': float(adjusted_rand_score(truth, labels)), 'silhouette': silhouette}


def align_labels(truth, predicted):
    """Hungarian matching for map colors only; ARI needs no label remapping.

    Unmatched predicted clusters get distinct new integers, never silently merge.
    """
    a, b = np.asarray(truth), np.asarray(predicted)
    if a.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ValueError('Labels must be matching nonempty vectors')
    true_values, true_indices = np.unique(a, return_inverse=True)
    pred_values, pred_indices = np.unique(b, return_inverse=True)
    table = np.zeros((len(true_values), len(pred_values)), dtype=int)
    np.add.at(table, (true_indices, pred_indices), 1)
    rows, cols = linear_sum_assignment(-table)
    mapping = {pred_values[c]: int(true_values[r]) for r, c in zip(rows, cols)}
    next_label = int(max(true_values)) + 1
    for value in pred_values:
        if value not in mapping:
            mapping[value] = next_label
            next_label += 1
    return np.array([mapping[value] for value in b])
