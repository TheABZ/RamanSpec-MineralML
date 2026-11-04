# Learning guide

Work through the lessons in order with the installed package. Each experiment is
synthetic. Keep a lab notebook recording parameters, observations and explanations;
changing a seed is an experiment, not a way to search for a flattering result.
Array conventions: rows are spectra/pixels; columns are spectral channels.

## Windows workflow

1. Install Python 3.11 or 3.12 and Git. Open PowerShell in your projects folder.
2. Run `git clone https://github.com/TheABZ/RamanSpec-MineralML.git` and
   `cd RamanSpec-MineralML`. If using the development branch before merge, run
   `git switch feat/synthetic-raman-research` after fetching it.
3. Run `py -3.12 -m venv .venv` (substitute 3.11 if installed).
4. Run `.venv\Scripts\python.exe -m pip install -e ".[dev]"`.
5. In PyCharm, open this repository. Under Settings → Project → Python Interpreter,
   add an existing local interpreter and select `.venv\Scripts\python.exe`.
   In VS Code use “Python: Select Interpreter” and choose the same file.
6. In the IDE terminal run `.venv\Scripts\python.exe -m ramanspec demo --output results`.
   A PyCharm run configuration can instead use module name `ramanspec`, parameters
   `demo --output results`, and the repository root as the working directory.
7. Open `results/reports/summary.md` and the PNGs in `results/figures/`.
8. Run `.venv\Scripts\python.exe -m pytest -q`.
9. Optional notebooks: install `-e ".[dev,notebooks]"`, launch
   `.venv\Scripts\python.exe -m jupyterlab`, and open `notebooks/`.
   Select the environment interpreter as the notebook kernel.

Activation is optional. Always use the environment's Python if there are multiple
installations. If a command cannot import `ramanspec`, check the interpreter and
that the editable installation ran from the repository root. No CUDA is needed.

## Lesson 1 — What a Raman spectrum represents

Light can scatter with a change in energy when interacting with molecular or
crystal vibrations. Raman shift is a difference in optical wavenumber, usually
reported in cm⁻¹. A spectrum plots a detector response against that shift. Peak
positions can carry structural information; intensities also depend on acquisition
conditions and orientation, so a peak is not automatically a unique mineral label.

**Equation:** Δν̃ = 1/λ₀ − 1/λs, where λ₀ is incident wavelength and λs is scattered
wavelength, both in cm. Positive Stokes shift corresponds to lower-energy scattered
light. This project simulates the resulting curve rather than light propagation.

**Code:** `simulate().axis` and `simulate().observed`; inspect
`notebooks/01_spectral_simulation.ipynb`.

**Exercise:** Print the first/last axis values and channel spacing. Plot one row
of `observed`. **Expected:** endpoints 400 and 1800; spacing 1400/600 ≈ 2.333 cm⁻¹;
intensity is arbitrary units, not photon counts calibrated to an instrument.

**Common error:** confusing wavelength in nm with Raman shift in cm⁻¹ or treating
class IDs as mineral names. **Self-check:** Why can two measurements of the same
material have different intensities? What calibration would real data require?

## Lesson 2 — Gaussian peak meaning

A Gaussian approximates a smooth localized peak; it is not a universal line shape.
Multiple peaks add to make one class signature.

**Equation:** I(ν)=A exp[−(ν−μ)²/(2σ²)]. A is maximum height, μ the center, and σ the
width. FWHM = 2√(2 ln 2)σ. Integrated area is Aσ√(2π), so changing width at fixed
height changes total area.

**Code:** `ramanspec.simulation.gaussian(axis, amplitude, center, width)`.

**Exercise:** On `np.linspace(400,1800,601)`, plot height 1, center 1000, widths
10 and 20. **Expected:** the second peak is broader with twice the continuous
area; sampled maximum may be below 1 if the center lies between channels.

**Common error:** using FWHM as σ or assuming sampled peaks always reach A.
**Self-check:** How would you hold peak area fixed while doubling σ? Why might a
Voigt profile be more appropriate for some future instrumental data?

## Lesson 3 — NumPy datasets and reproducibility

A matrix organizes repeated measurements compactly. Broadcasting combines a
per-spectrum coefficient of shape (900,1) with a spectral trend of shape (601,).
Randomness is controlled through a local generator rather than global state.

**Equation:** X ∈ ℝ^(N×M); Xᵢⱼ is channel j of sample i. Here N=900 and M=601.
The observation model is Y=S+B+E+C for clean signal, baseline, noise and spikes.

**Code:** `SimulationConfig`, `simulate`, and `SyntheticDataset.save`.

```python
import numpy as np
from ramanspec.simulation import simulate, SimulationConfig
first = simulate(SimulationConfig(seed=42))
second = simulate(SimulationConfig(seed=42))
assert np.array_equal(first.observed, second.observed)
assert np.allclose(first.observed, first.clean + first.baseline + first.noise + first.spikes)
```

**Exercise:** Compare seeds 42 and 43, then inspect `coordinates[31]`.
**Expected:** same-seed arrays match; changed seeds alter measurements; pixel 31
is row 1, column 1 in zero-based row-major order.

**Common error:** transposing the matrix so PCA treats channels as samples.
**Self-check:** Which dimension should normalization reduce? Why save parameters
as well as a seed?

## Lesson 4 — Baselines and nuisance structure

A slowly changing background can overwhelm weak peaks. Removing it is an
estimation problem because a measured spectrum does not reveal its components.

**Equation:** y(ν)=s(ν)+b(ν)+ε(ν)+c(ν); s is useful signal, b is baseline, ε is
measurement noise and c is sparse artifact intensity. Our baseline uses
b(t)=a₀+a₁t+a₂t²+a₃exp(−2t) with t∈[0,1].

**Code:** `simulate`, `estimate_baselines`, `preprocess`.

**Exercise:** Plot `clean[0]`, `baseline[0]` and `observed[0]`. Compare raw PCA with
processed PCA. **Expected:** the observed curve has raised/curved background and
noise; raw PCA can encode that nuisance variation. Exact component patterns
should be measured rather than assumed.

**Common error:** subtracting the known simulation baseline as the analysis
algorithm. That would leak unavailable truth into an apparently realistic pipeline.
**Self-check:** Why is truth useful for evaluation but inappropriate for preprocessing?

## Lesson 5 — AsLS and regularization

A baseline should follow a slowly varying background while ignoring most positive
peaks. AsLS alternates a smooth fit and asymmetric weighting.

**Equations:** J(z)=Σᵢwᵢ(yᵢ−zᵢ)²+λ||Dz||² and
(W+λDᵀD)z=Wy. z is the estimated baseline; W=diag(w); D computes second
differences; λ controls smoothness. Differentiate the quadratic objective with
fixed weights to obtain the linear system. Reweight with p above z and 1−p below.

**Code:** `asls(y, lam=1e5, p=.01, return_info=True)` and
`notebooks/02_baseline_correction.ipynb`.

**Exercise:** Sweep λ over 10⁴,10⁵,10⁶ and p over 0.001,0.01,0.05, using despiked
data. Calculate `rmse(true_baseline, estimate)`.
**Expected:** nine values; neither the largest λ nor smallest p is guaranteed best.
The full demo writes this table and a sensitivity plot. Check `info['converged']`.

**Common error:** taking a dense matrix inverse, forgetting that resampling changes
D's physical meaning, or equating convergence with accuracy. **Self-check:** What
happens as λ becomes very large? How is this analogous to smoothness-constrained
resistivity inversion? Why is λ not a physical mineral property?

## Lesson 6 — Cosmic-ray-like spike detection

A narrow positive artifact can be large enough to dominate PCA. A median is less
sensitive to isolated extremes than a mean. MAD measures a robust spread.

**Equation:** MAD(r)=median|r−median(r)|; σ̂≈1.4826 MAD for ideal Gaussian data.
Here r is the local-median residual. The detector also estimates scale from adjacent
differences divided by √2 and uses the larger estimate to protect noisy smooth
peaks. Candidates exceed eight scale units and have run length at most two.

**Code:** `despike`, `spike_metrics` and `tests/test_despiking.py`.

**Exercise:** Add a height-2 spike to a smooth peak-plus-noise spectrum. Compare the
mask to the injected location. Repeat with a genuine one-channel spectral peak.
**Expected:** isolated large spikes are repaired; a one-channel true peak may also
be removed. The tests explicitly expose this ambiguity rather than hiding it.

**Common error:** reporting recall without false positives, or lowering the threshold
until every spike is found. **Self-check:** What extra information would repeated
exposures provide? Why are interpolation endpoints less reliable?

## Lesson 7 — PCA, SVD, scores and loadings

PCA finds orthogonal directions capturing maximum variance in centered data. It
reduces dimensions without labels; it does not discover a chemical identity by itself.

**Equations:** Xc=UΣVᵀ and T=XcVₖ. U contains sample-side singular vectors; Σ contains
singular values; V contains spectral directions. Columns of T are scores; columns
of V (rows of scikit-learn's `components_`) are loadings. Explained variance of
component j is σⱼ²/(N−1); variance ratios divide by total variance.

**Code:** `run_pca(processed, n_components=5)` and notebook 03.

**Exercise:** Inspect shapes of scores and components. Plot PC1/PC2 and PC1/PC3;
compare loadings to spectral peaks. **Expected:** (900,5) scores and (5,601)
loadings for the default data; five ratios whose sum is at most 1.

**Common error:** calling a high-loading channel a definitive mineral marker, or
mistaking a sign flip for a change in physical meaning. **Self-check:** Why does
channel standardization change PCA? Can high explained variance be unwanted noise?

## Lesson 8 — k-Means and Gaussian mixtures

k-Means partitions score vectors by distance to centroids. GMM assigns probability
under a mixture of Gaussian distributions and can describe elliptical groups.

**Equations:** J=ΣₖΣᵢ∈Cₖ||xᵢ−μₖ||²; p(x)=ΣₖπₖN(x|μₖ,Σₖ).
Cₖ is a cluster, μₖ its center, πₖ a mixing weight and Σₖ a covariance matrix.
GMM uses expectation–maximization: update responsibilities, then parameters.

**Code:** `cluster(scores, n_clusters=3, seed=42)`; `model_selection(scores)`;
notebook 04.

**Exercise:** Compare K=2,3,4,5. Inspect inertia and GMM AIC/BIC. Confirm GMM
probabilities sum to one per sample. **Expected:** inertia generally decreases with
K; information criteria include a complexity penalty. A lower criterion is a
modeling preference, not proof of the correct number of minerals.

**Common error:** comparing cluster ID 0 directly to true class 0 or treating a 0.99
GMM assignment as 99% validated mineral confidence. **Self-check:** Why can two
identical partitions have different integer labels? What does covariance regularization do?

## Lesson 9 — Validation and honest interpretation

A model can look persuasive while being wrong. Use separate metrics for each
stage and keep assumptions visible.

**Equations:** RMSE=√mean((b−b̂)²), precision=TP/(TP+FP), recall=TP/(TP+FN),
FPR=FP/(FP+TN). ARI=(RI−E[RI])/(max(RI)−E[RI]) adjusts pair agreement for chance.
Silhouette for sample i is (bᵢ−aᵢ)/max(aᵢ,bᵢ), using within-cluster distance aᵢ
and nearest-other-cluster distance bᵢ.

**Code:** `validation.py`, `python -m pytest -q`, and generated metric tables.

**Exercise:** Permute cluster labels and recompute ARI. Increase noise; record
metrics for several seeds in your own experiment. **Expected:** ARI is unchanged by
ID permutation; noisy runs can degrade, but a single seed need not show a perfectly
monotonic trend. Quantify uncertainty before generalizing.

**Common error:** choosing parameters using truth and then describing scores as
independent validation. **Self-check:** What does a perfect synthetic ARI establish?
What would an independent real-data evaluation need?

## Lesson 10 — Connection to inverse problems and research

ERI/VES interpretation estimates subsurface properties from indirect observations.
Spectral baseline estimation likewise separates components using constraints.
Both require numerical stability, explicit assumptions and uncertainty analysis.
This is a methodological connection; the physical forward models are different.

**Equation:** a common inverse-problem objective is
||Wᵈ(d−F(m))||²+λ||Lm||². d denotes observations, F a forward model, m model
parameters, Wᵈ data weighting and L a regularizer. Compare its roles with y,z,W,D
in AsLS, while noting that AsLS has residual-dependent asymmetric weights.

**Code:** revisit `baseline.py` and `pipeline.py`, then read the interview guide.

**Exercise:** Write a paragraph separating numerical validation, physical validation
and geological interpretation. Design a future Raman–EDS comparison with spatial
registration and reference spectra. **Expected:** a plan that recognizes complementary
vibrational and elemental evidence and the need for independent calibration.

**Common error:** claiming this simulation measures mineral composition or partition
coefficients. **Self-check:** Which skills transfer from your groundwater study?
Which new spectroscopy skills still require training and experimental supervision?
