import numpy as np
import pytest
from ramanspec.clustering import cluster, model_selection
from ramanspec.dimensionality import run_pca
from ramanspec.simulation import simulate, SimulationConfig
from ramanspec.preprocessing import preprocess
from ramanspec.validation import evaluate_clusters, align_labels, spike_metrics, rmse


def test_pca_clustering_recovery():
    ds = simulate(SimulationConfig(rows=6, cols=9, channels=301))
    data, _ = preprocess(ds.observed)
    scores, pca = run_pca(data, 5)
    assert scores.shape == (54, 5)
    assert pca.components_.shape == (5, 301)
    result = cluster(scores)
    for name in ('kmeans', 'gmm'):
        assert result[name].shape == (54,)
        assert evaluate_clusters(ds.labels, result[name], scores)['adjusted_rand_index'] > .9
    np.testing.assert_allclose(result['probabilities'].sum(axis=1), 1)
    again = cluster(scores)
    np.testing.assert_array_equal(result['kmeans'], again['kmeans'])
    assert model_selection(scores, (2, 3)).shape == (2, 4)


def test_metrics_and_permutation():
    truth = np.array([0, 0, 1, 1, 2, 2])
    predicted = np.array([2, 2, 0, 0, 1, 1])
    np.testing.assert_array_equal(align_labels(truth, predicted), truth)
    scores = np.array([[0, 0], [0, .1], [3, 3], [3, 3.1], [9, 9], [9, 9.1]])
    assert evaluate_clusters(truth, predicted, scores)['adjusted_rand_index'] == 1
    assert rmse([0, 0], [3, 3]) == 3
    metrics = spike_metrics([1, 0, 1, 0], [1, 1, 0, 0])
    assert metrics['precision'] == metrics['recall'] == metrics['false_positive_rate'] == .5


def test_invalid_models():
    with pytest.raises(ValueError):
        run_pca(np.zeros((3, 5)), 2)
    with pytest.raises(ValueError):
        cluster(np.ones((5, 2)), 3)
    with pytest.raises(ValueError):
        run_pca(np.eye(4), 5)
