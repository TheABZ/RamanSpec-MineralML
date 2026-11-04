# Executed validation record

Validation was performed in the cloud workspace on Python 3.12.14. Tested direct
library versions are pinned in `requirements-tested.txt`; exact simulation and
analysis settings are preserved in `docs/metrics/provenance.json`.

## Commands executed

The environment interpreter was `/workspace/ramanspec-venv/bin/python`. Scientific
packages were inherited from the provided environment; pytest was installed through
the configured package proxy with the required execution permission.

```bash
python -m pip install --no-build-isolation --no-deps -e .
python -m pip wheel --no-deps --no-build-isolation . -w /tmp/ramanspec-wheels
python -m pytest -q
python -m ramanspec demo --output results
python -m ramanspec demo --output /tmp/ramanspec-repeat
```

The editable installation and wheel build succeeded. BLAS/OpenMP threads were
limited to one for validation (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`) and
`MPLCONFIGDIR` used a writable temporary directory. These are cloud execution
settings, not requirements for a normal Windows installation.

Final tests: **28 passed**. An initial run had 27 passes and one failing
peak-preservation test. Investigation showed the median residual underestimated
noise near a smooth peak. Adding a robust adjacent-difference noise estimate fixed
the tested failure; the assertion was preserved. The final run emitted one
external anyio pytest-plugin deprecation warning; no project test was skipped.

All code cells in all four notebooks were executed sequentially in separate Python
namespaces with the installed package. This checks their code; it is not a claim
that an interactive Jupyter frontend was tested. Windows and Python 3.11 validation
are configured in GitHub Actions but were not executed in this local environment.

## Default experiment: measured results

| Quantity | Observed value |
|---|---:|
| Spectra × channels | 900 × 601 |
| k-Means ARI | 1.0 |
| GMM ARI | 1.0 |
| Silhouette (both partitions) | 0.9117569433 |
| Baseline RMSE | 0.0314262852 a.u. |
| AsLS converged | 900 / 900, at most 9 iterations |
| Spike true positives | 1115 |
| Spike false positives / false negatives | 0 / 0 |
| Spike precision / recall | 1.0 / 1.0 |
| Clean-channel false-positive rate | 0.0 |
| Dominant clean peaks preserved within 1% | 100% |
| First two PCs: total variance fraction | 0.9425836090 |

Exact JSON and comparison tables are retained under `docs/metrics/` as small
reviewable evidence. Each full run also writes processed data, ground truth,
probabilities and predictions under its output directory.

Both models retained ARI=1 across noise levels 0.01, 0.05, 0.15 and 0.30, while
silhouette declined from approximately 0.914 to 0.703. This experiment therefore
shows reduced separation, not measured classification failure. It does not locate
the noise level where recovery fails. GMM AIC and BIC favored K=3 among K=2–5.
These are single-seed observations on deliberately distinct synthetic signatures.

## Reproducibility and plots

Two complete default runs produced exactly equal metrics JSON, all metric CSV
bytes, and every array in `spectra.npz` and `analysis.npz`. Cross-platform bitwise
identity is not guaranteed. Each run generated 13 PNG figures. Selected baseline,
map and noise figures are retained in `docs/examples/`; all figures are available
locally in `results/figures/` and can be regenerated.

## Interpretation limits

Perfect spike and partition metrics here do not certify real spectral data.
Unresolved genuine peaks are a demonstrated detector failure mode. Baseline bias
remains nonzero, and five PCs discard variance. No independent experimental or
held-out classification benchmark, uncertainty interval, or mineral composition
measurement was performed. See `RESEARCH_METHODS.md` for assumptions and future work.
