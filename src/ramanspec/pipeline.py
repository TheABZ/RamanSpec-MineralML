"""Experiment orchestration and auditable exports, separate from scientific kernels."""
from dataclasses import asdict, replace
from importlib.metadata import version
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
from .simulation import SimulationConfig, simulate
from .preprocessing import preprocess
from .baseline import asls
from .despiking import despike
from .dimensionality import run_pca
from .clustering import cluster, model_selection
from .validation import rmse, spike_metrics, evaluate_clusters


def run_demo(output='results', config=None, n_components=5, n_clusters=3, experiments=True):
    """Run the default 900-spectrum experiment; export all numerical evidence.

    Ground truth is not passed to preprocess, PCA, or either clustering model.
    The default K=3 is a known synthetic-design assumption, not inferred geology.
    """
    config = config or SimulationConfig()
    if n_components < 3:
        raise ValueError('The demo needs at least three PCs for its figures')
    output = Path(output)
    for name in ('figures', 'metrics', 'reports', 'data'):
        (output / name).mkdir(parents=True, exist_ok=True)
    dataset = simulate(config)
    dataset.save(output / 'data')
    processed, diagnostics = preprocess(dataset.observed)
    scores, pca = run_pca(processed, n_components, config.seed)
    clusters = cluster(scores, n_clusters, config.seed)
    metrics = {name: evaluate_clusters(dataset.labels, clusters[name], scores, config.seed) for name in ('kmeans', 'gmm')}
    metrics['baseline_rmse'] = rmse(dataset.baseline, diagnostics['baseline'])
    metrics['spikes'] = spike_metrics(dataset.spike_mask, diagnostics['spike_mask'])
    clean_despiked, clean_flags = despike(dataset.clean)
    peak_channels = np.argmax(dataset.clean, axis=1)
    peak_relative_error = np.abs(clean_despiked[np.arange(len(peak_channels)), peak_channels] - dataset.clean.max(axis=1)) / dataset.clean.max(axis=1)
    metrics['peak_preservation'] = {'clean_signal_rmse': rmse(dataset.clean, clean_despiked),
        'clean_channel_false_positive_rate': float(clean_flags.mean()),
        'mean_dominant_peak_relative_error': float(peak_relative_error.mean()),
        'dominant_peaks_within_1_percent': float((peak_relative_error <= .01).mean())}
    metrics['explained_variance_ratio'] = pca.explained_variance_ratio_.tolist()
    convergence = [asls(row, return_info=True)[1] for row in diagnostics['despiked']]
    metrics['asls_convergence'] = {'converged': sum(info['converged'] for info in convergence),
                                 'total': len(convergence), 'maximum_iterations': max(info['iterations'] for info in convergence)}
    # Ground truth optimizes no pipeline hyperparameter: this is a diagnostic only.
    indices = np.linspace(0, len(dataset.labels)-1, min(30, len(dataset.labels)), dtype=int)
    sensitivity = []
    for lam in (1e4, 1e5, 1e6):
        for p in (.001, .01, .05):
            estimates = np.stack([asls(diagnostics['despiked'][i], lam=lam, p=p) for i in indices])
            sensitivity.append({'lambda': lam, 'p': p, 'baseline_rmse': rmse(dataset.baseline[indices], estimates)})
    sensitivity = pd.DataFrame(sensitivity)
    selection = model_selection(scores, counts=tuple(k for k in (2, 3, 4, 5) if k < len(scores)), seed=config.seed)
    noise_rows = []
    # Same seed and shapes give paired scenarios with scaled measurement noise.
    for noise in ((.01, .05, .15, .30) if experiments else (config.noise,)):
        if noise == config.noise:
            ds, nscores, result = dataset, scores, clusters
        else:
            ds = simulate(replace(config, noise=noise))
            nscores, _ = run_pca(preprocess(ds.observed)[0], n_components, config.seed)
            result = cluster(nscores, n_clusters, config.seed)
        for name in ('kmeans', 'gmm'):
            noise_rows.append({'noise': noise, 'model': name, **evaluate_clusters(ds.labels, result[name], nscores, config.seed)})
    noise_results = pd.DataFrame(noise_rows)
    sensitivity.to_csv(output / 'metrics/baseline_sensitivity.csv', index=False)
    noise_results.to_csv(output / 'metrics/noise_comparison.csv', index=False)
    selection.to_csv(output / 'metrics/model_selection.csv', index=False)
    (output / 'metrics/metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    predictions = pd.DataFrame({'row': dataset.coordinates[:, 0], 'column': dataset.coordinates[:, 1],
        'synthetic_truth': dataset.labels, 'kmeans': clusters['kmeans'], 'gmm': clusters['gmm']})
    for k in range(n_clusters):
        predictions[f'gmm_probability_{k}'] = clusters['probabilities'][:, k]
    predictions.to_csv(output / 'metrics/predictions.csv', index=False)
    np.savez_compressed(output / 'data/analysis.npz', processed=processed, scores=scores,
                        loadings=pca.components_, baseline=diagnostics['baseline'], spike_mask=diagnostics['spike_mask'])
    versions = {name: version(name) for name in ('numpy', 'scipy', 'pandas', 'matplotlib', 'scikit-learn', 'ramanspec-mineralml')}
    provenance = {'python': platform.python_version(), 'libraries': versions, 'simulation': asdict(config),
                  'pca_components': n_components, 'clusters': n_clusters, 'normalize': True, 'standardize': False,
                  'asls': {'lambda': 1e5, 'p': .01, 'max_iter': 30, 'tol': 1e-5},
                  'despiking': {'window': 7, 'threshold': 8., 'max_width': 2}, 'noise_experiments': experiments}
    (output / 'reports/provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    from .visualization import make_figures
    make_figures(dataset, diagnostics, scores, pca, clusters, sensitivity, noise_results, selection, output / 'figures')
    report = ['# Synthetic Raman experiment', '', 'Educational simulation only; no experimental mineral identification.', '',
              f'Samples: {len(dataset.labels)}; channels: {config.channels}; seed: {config.seed}.',
              f'Baseline RMSE: {metrics["baseline_rmse"]:.6f} a.u.',
              f'AsLS converged for {metrics["asls_convergence"]["converged"]}/{len(dataset.labels)} spectra.']
    for name in ('kmeans', 'gmm'):
        report.append(f'{name}: ARI = {metrics[name]["adjusted_rand_index"]:.6f}; silhouette = {metrics[name]["silhouette"]}.')
    report += ['', '## Spike detection', '', json.dumps(metrics['spikes'], indent=2), '',
               '## Interpretation and limits', '',
               'ARI evaluates agreement with invented classes and is invariant to cluster-number permutations. '
               'Map colors alone use truth-based Hungarian alignment for display. '
               'GMM probabilities are conditional model assignments, not calibrated mineral confidence. '
               'Baseline sensitivity uses 30 evenly spaced spectra and does not tune the pipeline. '
               'Noise comparisons use one paired seed, so they do not establish uncertainty or statistical significance. '
               'Broad peaks are protected heuristically; unresolved true peaks can be mistaken for spikes. '
               'See metrics/*.csv, metrics.json and provenance.json for exact numerical evidence.']
    (output / 'reports/summary.md').write_text('\n\n'.join(report) + '\n', encoding='utf-8')
    return metrics
