# Interview preparation

Use these as explanations to practice and adapt after running the code yourself.
Only describe work you understand and can demonstrate. This independent educational
project is relevant preparation for computational research; it does not establish
experimental experience or results for a proposed PhD.

## A 60-second explanation

“I built a Python demonstration to understand how spectral preprocessing affects
unsupervised classification. It simulates 900 Raman-like spectra in three invented
classes on a 30-by-30 grid, adds baseline interference, noise and spike artifacts,
and preserves the true components for evaluation. I remove spikes using robust
statistics, estimate baselines using a sparse regularized solver, and compare
k-Means with Gaussian mixtures after PCA. The code exports figures and metrics and
has automated tests. My ERI/VES background made the regularization and inverse
problem aspects particularly interesting. The main lesson is that a strong
classification score depends on preprocessing and assumptions. These are synthetic
results, so they demonstrate a computational workflow, not validated mineral
identification.”

## A three-minute technical explanation

“Each spectrum is a sum of Gaussian peaks with controlled variation. I add a smooth
background, Gaussian measurement noise and sparse positive artifacts. Synthetic
truth lets me evaluate baseline error and spike precision separately from cluster
recovery. The spatial grid makes the results interpretable, but coordinates do not
enter the clustering algorithm.

“For despiking, I use positive deviations from a local median, robust noise scales,
and a width restriction before interpolation. A test revealed that a residual-only
noise estimate could remove a noisy genuine peak. Adding a scale from adjacent
channel differences addressed that tested case. A one-channel true peak is still
ambiguous; this is a documented limitation.

“AsLS balances data fit with a second-difference smoothness penalty. With fixed
weights I solve a sparse linear system, then downweight positive residuals to avoid
following peaks. I check both numerical convergence and error against the known
baseline; those are different concepts. I examine sensitivity to smoothness and
asymmetry without feeding the best truth-based setting into classification.

“After vector normalization, PCA summarizes the centered spectra. Scores describe
samples and loadings describe spectral directions. I compare k-Means, which uses
centroid distances, with full-covariance Gaussian mixtures. I report ARI against
synthetic labels, silhouette, information criteria and paired noise experiments.
Integer cluster labels are arbitrary, so map colors use post-hoc alignment only.

“The experiment makes reproducibility and limitations explicit: seeds, parameters,
versions and results are exported. The next research step would be validated
experimental data with acquisition metadata, repeated measurements and independent
reference assignments. My resistivity experience motivates the emphasis on
regularization and uncertainty, while spectroscopy introduces new physical and
experimental questions.”

## Mathematical methods you should be able to explain

1. **Gaussian signal:** height A, center μ, width σ; distinguish width from integrated
   area and connect σ to FWHM. Explain why real peaks may require other shapes.
2. **AsLS derivation:** differentiate `(y−z)ᵀW(y−z)+λzᵀDᵀDz` with fixed W to obtain
   `(W+λDᵀD)z=Wy`. Explain sparse storage, weight updates, convergence, and λ sensitivity.
3. **PCA:** center data, factor `Xc=UΣVᵀ`, and compute scores `T=XcVₖ`. Variance is
   not automatically chemical relevance; signs of loadings are arbitrary.
4. **k-Means:** alternating assignment and centroid updates minimize squared distances
   to a local optimum. Multiple initializations reduce initialization sensitivity.
5. **GMM:** latent memberships and EM, mixing weights summing to one, covariance
   matrices, and regularization against singular components. Distinguish likelihood,
   model selection and calibrated confidence.
6. **Validation:** RMSE for continuous estimates; precision/recall/FPR for spikes;
   ARI for permutation-invariant partitions. Explain why silhouette alone is insufficient.

## Technical questions and grounded answers

**Why synthetic data?** It provides known clean signal, baseline, spike locations
and labels, making stage-specific error measurable. It cannot reproduce every
instrument or natural material effect.

**Why sparse matrices?** The second-difference operator has only three diagonals.
A dense inverse would waste memory and computation; a sparse solve exploits that
structure and avoids explicitly forming an inverse.

**Why p=0.01?** It favors a lower smooth envelope because positive peaks receive
less weight. It is an educational default checked by sensitivity analysis, not a
universal setting. Too small p can underestimate a noisy baseline.

**How did testing change the implementation?** A noisy resolved peak was altered
by about 0.019 a.u. under the original detector. A second robust scale from channel
differences protected that case. The unchanged test was rerun after the correction;
the project also tests an unresolved-peak failure mode explicitly.

**Why not use labels during preprocessing?** Experimental truth is usually unavailable.
Using it would produce optimistic analysis. Labels are used only for scoring and
figures; K=3 is explicitly assumed from the simulation design.

**Are probabilities reliable confidence estimates?** They are GMM responsibilities
conditional on a model and the retained PCA representation. They are not calibrated
probabilities of mineral identity or evidence of chemical composition.

**How good are the results?** Consult `docs/validation.md` for the actual executed
numbers. Explain that high scores on intentionally separable synthetic classes
are a software/numerical demonstration, not a real-data benchmark.

**How would you test generalization?** Use independently acquired, calibrated spectra
with verified labels, instrument/session separation, fitted preprocessing reused on
held-out data, and multiple runs or bootstrap uncertainty appropriate to sampling.
Do not call the current in-sample partition comparison held-out validation.

**What can go wrong?** Unresolved peaks can resemble spikes; broad peaks and baselines
can overlap; normalization removes absolute intensity information; PCA can retain
nuisance variation; model assumptions and K can be wrong.

**How does ERI/VES connect?** Both require indirect inference, regularization,
sensitivity analysis and careful distinction between model fit and physical truth.
Electrical resistivity forward physics differs from light–matter interaction, so
shared numerical skills do not replace spectroscopy training.

**How might Raman and EDS complement each other?** Raman probes vibrational/structural
features, while EDS provides elemental information subject to its own limitations.
Spatial registration, acquisition metadata, reference materials and uncertainty
would be needed to compare them meaningfully. No Raman–EDS data are implemented here.

## Relevance to future PhD work

For a potential Manchester interview on metal transport and mineralization in
igneous systems, emphasize reproducible computation, quantitative analysis,
numerical reasoning and a willingness to learn experimental methods. Discuss
how validated mineral characterization could eventually support broader geological
questions. Do not imply that this simulation has measured fluid–melt partition
coefficients, validated mineral compositions, established a transport mechanism,
or generated conclusions endorsed by prospective supervisors.

The project demonstrates modular Python, sparse numerical methods, testing,
visualization and honest evaluation. It does not demonstrate instrument operation,
experimental calibration, natural sample interpretation, or independent geological
discovery. A useful interview demonstration is to change one parameter, predict its
possible effect, run the code, and explain both the observed result and its limits.
