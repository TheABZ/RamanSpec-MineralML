# Research methods

## Motivation and problem definition

This educational computational study asks whether three invented spectral
populations can be recovered after simulated baseline, noise and spike corruption.
Known components permit evaluation of preprocessing error independently from
clustering agreement. It is not an experimental mineralogical study.

## Synthetic dataset design

`SimulationConfig` defaults to 30 rows × 30 columns, 601 channels spanning
400–1800 cm⁻¹, seed 42, nominal noise standard deviation 0.025 a.u., and an
independent spike probability of 0.002 per channel. Spatial labels occupy wavy
vertical regions. No spatial coordinates enter model fitting.

For each sample, the signal is

\[s(\nu)=\sum_j A_j\exp[-(\nu-\mu_j)^2/(2\sigma_j^2)].\]

Here ν is Raman shift, A is peak height, μ is center, and σ is width in cm⁻¹.
The full width at half maximum is `2 sqrt(2 ln 2) σ`. Three invented signatures
use centers (600, 1000, 1450), (650, 1005, 1350), and (610, 1120, 1500) cm⁻¹.
These numbers are design choices, **not mineral reference assignments**.
Amplitudes vary uniformly by ±15%, widths by ±10%, and centers by normal jitter
with standard deviation 2 cm⁻¹. See `simulation.py` for amplitudes and widths.

Let t be the shift rescaled to [0,1]. The baseline is a positive intercept plus
linear and quadratic terms and a decaying exponential. Coefficients vary per
spectrum. Independent normal noise has a per-spectrum scale factor between 0.8
and 1.2. Positive spikes have independent uniform amplitudes from 1 to 2.5 a.u.
Adjacent spikes are possible. All components, masks, coordinates, labels and
parameters are exported separately, satisfying `observed = clean + baseline +
noise + spikes` up to floating-point rounding.

## Despiking

For a seven-channel window, let mᵢ be the reflected local median and rᵢ=yᵢ−mᵢ.
Estimate a scale separately for each spectrum as

\[\hat\sigma=1.4826\operatorname{median}|r-\operatorname{median}(r)|.\]

The implemented scale is the larger of this residual MAD and the scaled MAD
of adjacent differences divided by √2. The latter counters noise underestimation
by the median residual near smooth peaks, assuming independent adjacent noise.
A numerical floor avoids division-like degeneracy in zero-noise spectra. Positive
residuals above `median(r) + 8 σ̂` are candidates. Only contiguous runs of at most
two channels are replaced by interpolation between surviving samples. At an
endpoint the nearest surviving value is used. Original arrays are copied.

The factor 1.4826 makes MAD consistent for ideal Gaussian samples; median-filter
residuals are not independent Gaussian noise, so the threshold is heuristic.
Wide candidate runs are retained to protect resolved peaks. True peaks narrower
than the instrument/channel spacing can be indistinguishable from spikes: a test
explicitly demonstrates that a one-channel true peak is removed. Neighboring
pixels or repeated measurements would provide stronger evidence in real work.

## Asymmetric least squares and regularization

For observed, despiked y, estimate a smooth baseline z by alternating linear
solution and residual-dependent weights:

\[J(z)=(y-z)^T W(y-z)+\lambda\|Dz\|^2,\]
\[D z=(z_0-2z_1+z_2,\ldots,z_{n-3}-2z_{n-2}+z_{n-1})^T.\]

With fixed W, setting the derivative to zero gives
`−2W(y−z)+2λDᵀDz=0`, hence `(W+λDᵀD)z=Wy`. D is (n−2) × n;
W is diagonal. Positive λ and strictly positive weights give a positive-definite
system. SciPy sparse diagonals and `spsolve` avoid a dense inverse.

Start with unit weights. After each solve use weight p above the baseline and
1−p at or below it. Default λ=100000, p=0.01, maximum 30 iterations. Stop only when
relative baseline change is below 1e−5 and weights are unchanged. Individual
calls can return convergence information; the demo reports converged counts.
Reaching the iteration limit is reported as nonconvergence, not success.

Increasing λ discourages curvature; decreasing p reduces the influence of positive
peaks but may bias estimates below the true baseline in noisy spectra. D is based
on channel indices, not a continuous derivative divided by physical spacing:
resampling requires reassessing λ. The simulation has uniform spacing.
Sensitivity analysis evaluates λ ∈ {10⁴,10⁵,10⁶} and p ∈ {0.001,0.01,0.05} on
30 evenly spaced samples against known baselines. It is a diagnostic; the best
truth-based parameter is not fed into the clustering pipeline.

## Preprocessing and PCA

Nonfinite samples raise an error by default. An explicit interpolation policy
fills missing values along channel indices, including nearest-value endpoint
extension, and requires at least two finite channels. Apply despiking, baseline
subtraction, then optional L2 normalization. Zero spectra remain zero. Negative
baseline-corrected intensities are retained rather than clipped.

Optional `StandardScaler` standardizes each channel across samples after vector
normalization. This can magnify weak noisy channels and changes the PCA metric;
it is disabled in the demo. A fitted scaler is returned for reuse on future data.

PCA centers each channel across samples. For centered Xc,

\[X_c=U\Sigma V^T,\quad T=X_cV_k.\]

Rows of `components_` are retained unit directions Vₖᵀ (loadings); rows of T are
sample coordinates (scores). The demo uses five components with deterministic
randomized SVD. Explained variance measures variance captured, not chemical
importance. Signs of components are arbitrary. Background or overall intensity
can dominate PCA unless preprocessing is appropriate. No labels enter PCA.

## Clustering and model diagnostics

k-Means minimizes `Σₖ Σᵢ∈Cₖ ||xᵢ−μₖ||²` in PCA score space, with ten initializations.
GMM models `p(x)=Σₖ πₖ N(x|μₖ,Σₖ)` using full covariance matrices, three
initializations, and covariance regularization 10⁻⁶. EM estimates mixture
parameters; responsibilities give conditional assignment probabilities. Failure
to converge raises an error. Both models use the simulation seed.

Default K=3 reflects the known simulation design. Runs at K=2–5 export inertia,
AIC and BIC. Lower AIC/BIC trades fit against parameter complexity; smaller
inertia alone cannot select K because it decreases as K grows. These diagnostics
apply to the fixed retained PCA space, not to all original spectral information.

## Validation methodology

- Baseline error: `sqrt(mean((true−estimated)²))` in a.u.
- Spike precision: TP/(TP+FP); recall: TP/(TP+FN); FPR: FP/(FP+TN).
  Undefined precision/recall use zero by explicit convention; raw counts accompany them.
- Peak preservation: rerun despiking on clean spectra, measure clean-channel FPR,
  signal RMSE, and dominant-peak relative amplitude error. The proportion within
  1% is reported. This does not certify every peak or every possible width.
- ARI adjusts pairwise partition agreement for chance and is invariant to cluster
  numbering. Hungarian alignment is used only for map colors.
- Silhouette compares within-cluster distances with nearest other-cluster distances
  on up to 500 deterministically sampled scores. It is undefined for one cluster
  or one cluster per sample and then exported as null.
- Paired noise scenarios use standard deviations 0.01, 0.05, 0.15 and 0.30 with
  the same seed, spatial arrangement and signatures. All 900 spectra are refitted.

All models are fitted and evaluated on this simulated dataset: ARI describes
partition recovery, not held-out predictive performance. Ground-truth components
are restricted to diagnostics and evaluation. Multi-seed replications and an
independent experimental test set would be needed for stronger conclusions.

## Observed results

The committed `docs/validation.md` records the commands, versions and measured
results from the executed build. `results/reports/summary.md` is generated by each
run, with exact settings in `provenance.json`. Metrics from a different configuration
must not be represented as the default result. No theoretical prediction is
substituted for an unexecuted experiment.

## Limitations and future development

Gaussian peaks omit Lorentzian/Voigt shapes, instrument response, calibration drift,
shot noise, fluorescence complexity, orientation effects and mixtures. Three
separable classes make clustering easier than natural assemblages. The spatial
map is synthetic and contributes no spatial regularization. A single seed provides
no confidence interval. Baseline truth is available only because this is simulated.

Extensions include parameter stability over seeds, mixed spectra, calibrated import
and reference matching, repeated-exposure despiking, instrument-specific validation,
uncertainty propagation and co-registered Raman–EDS analysis. EDS elemental evidence
and Raman vibrational evidence are complementary but require reference materials,
registration and independent interpretation.

## Verifiable reference resources

- NumPy random generator: https://numpy.org/doc/stable/reference/random/generator.html
- SciPy sparse solver: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html
- scikit-learn PCA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- scikit-learn mixture models: https://scikit-learn.org/stable/modules/mixture.html
- scikit-learn clustering evaluation: https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation

These are implementation references, not sources for mineral peak assignments.
