# Development Addendum: Local History Information Is Not Risk Certification

## Experiment

We replaced a shared risk forest with an exact source-excluded local estimator.
The estimator used the same 120 causal features, fitting-only standardization,
two signed risk targets and original source/query weights. Each query selected
the nearest window in each conservative source/agent identity bucket and then
the nearest 32 buckets. The entire query source was excluded from all fitted
preprocessing and references. No query future label entered retrieval.

All 48,061 known proposals were evaluated in eight exposed development folds,
grouped into three producer contexts. The 1,274 unknown proposals were retained
in the audit but not assigned invented supervision. The comparison included
the verified shared64 forest, a fitting-prior mean and one fixed within-source
query-reference permutation. No threshold, neighbor-count or source search
was performed. This is not an independent test or a novel nearest-neighbor method.

## Results

Relative to shared64, source-macro decision-cost reductions were -4.482%,
-121.691% and -2.568%; harmful admissions increased from 0/1/0 to 107/723/521.
The internal information screen failed 0/3. Context2 normalized risk MSE
improved by 33.420%, although a constant fitting prior already improved it by
30.094%. Correct matching reduced cost versus the fixed misalignment by
11.453%, 20.948% and 15.575%. This establishes neither statistical significance
nor compliance with the unchanged full-policy risk and support contract.

Source076 showed severe conditional mismatch even among nearby histories:
predicted signed-risk exceedance was 6.535% versus 31.409% observed in the
[0.25,0.5) nearest-distance bin, and 8.243% versus 48.334% in [0.5,1).
Distances are standardized feature distances, not physical distance. These
observations motivate representation and source-support analysis, not a
posthoc threshold or an assertion of irreducible uncertainty.

## Position Relative To Risk-Control Literature

Conformal Risk Control extends split conformal calibration to expected
monotone losses. Our fitted local risk averages are not that calibration
procedure and do not inherit its guarantee merely by predicting a risk.
[Angelopoulos et al., Conformal Risk Control](https://arxiv.org/abs/2208.02814).

Learn then Test frames calibration as multiple hypothesis testing. The single
misalignment comparison here is a development control, not a completed Learn
then Test calibration or a family-wise risk certificate.
[Angelopoulos et al., Learn then Test](https://arxiv.org/abs/2110.01052).

The present task additionally keeps the original per-view positive-harm,
easy-net and reference-support requirements. No implication from a favorable
average loss to every-view compliance is claimed. Empty support stays NA;
identity fallback is not a new exemption. Our method contribution would need
to establish useful joint intervention under that contract, which remains
unproven. No new theory or submission-ready claim follows from this addendum.

## Limits And Reproduction

Only already-exposed own-B data were used. Overlapping windows, weak future
labels and few donor sources limit generalization; conservative local-ID
buckets do not certify distinct physical people. All eight retrievals were
recomputed exactly with a different chunk size. Exact reproduction is an
engineering result, not independent scientific confirmation.

Native obs8/pred12 image-coordinate silver data remain non-metric, with no
seconds-level, true3D, foundation or generative claim. Stage5C/SMC are off.
The compiled manuscript is unchanged; this standalone addendum records a
negative development result rather than replacing the main empirical claim.
