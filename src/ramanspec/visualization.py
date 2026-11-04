"""Headless scientific figures; class labels are explicitly synthetic."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .validation import align_labels


def save_figure(fig, path):
    """Save at 180 dpi and close to bound memory during batch experiments."""
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def make_figures(dataset, diagnostics, scores, pca, clusters, sensitivity, noise_results, selection, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    axis = dataset.axis
    examples = [np.flatnonzero(dataset.labels == k)[0] for k in np.unique(dataset.labels)]
    for name, spectra in [('clean_spectra', dataset.clean), ('observed_spectra', dataset.observed)]:
        fig, ax = plt.subplots(figsize=(8, 4))
        for i in examples:
            ax.plot(axis, spectra[i], label=f'Synthetic class {dataset.labels[i]}', lw=1)
        ax.set(xlabel='Raman shift (cm⁻¹)', ylabel='Intensity (a.u.)', title=name.replace('_', ' ').capitalize())
        ax.legend()
        save_figure(fig, output / f'{name}.png')
    i = int(np.argmax(dataset.spike_mask.sum(axis=1)))
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for values, label in [(dataset.observed[i], 'Observed'), (dataset.clean[i], 'Clean signal'),
                          (dataset.baseline[i], 'True baseline'), (diagnostics['baseline'][i], 'AsLS baseline')]:
        axes[0].plot(axis, values, label=label, lw=1)
    axes[1].plot(axis, diagnostics['despiked'][i] - diagnostics['baseline'][i], label='Corrected', lw=1)
    axes[1].plot(axis, dataset.clean[i], label='Clean signal', lw=1)
    for ax in axes:
        ax.set(ylabel='Intensity (a.u.)')
        ax.legend()
    axes[0].set_title('Baseline correction: synthetic ground-truth comparison')
    axes[1].set_xlabel('Raman shift (cm⁻¹)')
    save_figure(fig, output / 'baseline_correction.png')
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(axis, dataset.observed[i], label='Observed', lw=1)
    ax.plot(axis, diagnostics['despiked'][i], label='Despiked', lw=1)
    flags = diagnostics['spike_mask'][i]
    ax.scatter(axis[flags], dataset.observed[i, flags], marker='x', color='red', label='Detected')
    ax.set(xlabel='Raman shift (cm⁻¹)', ylabel='Intensity (a.u.)', title='Positive spike detection')
    ax.legend()
    save_figure(fig, output / 'despiking.png')
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    pcs = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    axes[0].bar(pcs, pca.explained_variance_ratio_)
    axes[1].plot(pcs, np.cumsum(pca.explained_variance_ratio_), 'o-')
    for ax in axes:
        ax.set(xlabel='Principal component', ylabel='Explained variance fraction', ylim=(0, 1.02))
    axes[0].set_title('Scree plot')
    axes[1].set_title('Cumulative explained variance')
    save_figure(fig, output / 'pca_variance.png')
    for name, labels in [('truth', dataset.labels), ('kmeans', clusters['kmeans']), ('gmm', clusters['gmm'])]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, component in zip(axes, [1, 2]):
            scatter = ax.scatter(scores[:, 0], scores[:, component], c=labels, s=8, cmap='tab10', vmin=0, vmax=9)
            ax.set(xlabel='PC1 score', ylabel=f'PC{component+1} score', title=f'{name}: synthetic spectra')
        fig.colorbar(scatter, ax=axes.tolist(), label='Class / cluster ID', ticks=np.unique(labels), shrink=.8)
        # Explicit spacing avoids tight_layout conflicts with the shared colorbar.
        fig.subplots_adjust(right=.80, wspace=.30)
        fig.savefig(output / f'pca_scores_{name}.png', dpi=180, bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    for j in range(min(3, pca.n_components_)):
        ax.plot(axis, pca.components_[j], label=f'PC{j+1}')
    ax.set(xlabel='Raman shift (cm⁻¹)', ylabel='Loading (unit vector coefficient)', title='PCA loadings')
    ax.legend()
    save_figure(fig, output / 'pca_loadings.png')
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    maps = [('Synthetic truth', dataset.labels)] + [(f'{name} (aligned)', align_labels(dataset.labels, clusters[name])) for name in ('kmeans', 'gmm')]
    for ax, (title, labels) in zip(axes, maps):
        ax.imshow(labels.reshape(dataset.config.rows, dataset.config.cols), cmap='tab10', vmin=0, vmax=9, interpolation='nearest')
        ax.set(title=title, xlabel='Column (pixel)', ylabel='Row (pixel)')
    save_figure(fig, output / 'spatial_maps.png')
    fig, ax = plt.subplots(figsize=(7, 4))
    for p, group in sensitivity.groupby('p'):
        ax.semilogx(group['lambda'], group['baseline_rmse'], 'o-', label=f'p={p}')
    ax.set(xlabel='AsLS λ', ylabel='Baseline RMSE (a.u.)', title='Sensitivity: diagnostic subset')
    ax.legend()
    save_figure(fig, output / 'baseline_sensitivity.png')
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, group in noise_results.groupby('model'):
        ax.plot(group.noise, group.adjusted_rand_index, 'o-', label=name)
    ax.set(xlabel='Noise standard deviation (a.u.)', ylabel='Adjusted Rand index', title='Synthetic noise experiment', ylim=(-.05, 1.05))
    ax.legend()
    save_figure(fig, output / 'noise_comparison.png')
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(selection.clusters, selection.inertia, 'o-')
    axes[0].set(ylabel='k-Means inertia')
    for key in ('aic', 'bic'):
        axes[1].plot(selection.clusters, selection[key], 'o-', label=key.upper())
    axes[1].legend()
    axes[1].set(ylabel='GMM information criterion')
    for ax in axes:
        ax.set(xlabel='Number of clusters', title='Model selection on PCA scores')
    save_figure(fig, output / 'model_selection.png')
