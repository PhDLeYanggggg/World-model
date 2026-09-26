# Fitting-Only Fixed-Cap Diagnostic

## Material Passport

Source: frozen risk-conditioned experiment and its checked ancestry.
Task: offline implementation/error diagnosis, not a new prediction method.
Scope: 144 source-development fitting views, three seeds, both full and
motion-only inputs; original in-sample, OOF and both cyclic inner controls.
New computation is fresh_run; frozen scores/targets are cached_verified.
No new model training, held readout, independent calibration or deployment.

## Question

How much of the existing fitting easy-harm MSE is the minimum realized-label
error allowed by the frozen all-harm prediction H? Is it still present when
the diagnostic box is enlarged to the causal disagreement envelope?

For prediction p, nonnegative target y and cap c, q=min(y,c):

    (p-y)^2 = (q-y)^2 + (p-q)^2 + 2(y-q)(q-p).

All terms are nonnegative when 0 <= p <= c. The first term is the empirical
projection floor of this fixed box, NOT irreducible prediction error or an
achievable model. q uses a realized future label and must never be an input,
feature, policy, deployable prediction or promoted result.

## Fixed Design

Use only the three meta-fitting localities, never the outer held target.
Keep the prior common-event target cut, all row IDs, teacher ancestry and
score hashes. The OOF predictor excludes the row's inner locality from its
fitting, but the meta target cut is defined on three fitting localities; this
is not independent calibration. Original/cyclic banks retain their truthful
in-sample status. No result-driven view, seed, cutoff or bank selection.

For each bank compute both unweighted positive-envelope means and the existing
registered risk-weighted means. Unknown labels remain excluded. Compare the
frozen H cap and causal envelope cap. Record direct MSE, floor share, remaining
distance, cross term, above-cap fraction and exact arithmetic identity.
This gives 144 x 4 x 2 x 2 = 2,304 exact identities.

Also record three weighted-tercile bins of the causal score H/envelope, fitted
only on fitting rows. Report mean(y-H) in each bin, normalized by target RMS
for scale description. These are resubstitution descriptive moments, not
independent conditional-bias tests, risk certificates or deployable bins.
Keep empty cells and zero denominators explicit; do not tune their boundaries.

Report all 144 dependent views, not an independent sample size. No bootstrap
of overlapping rows and no new significance/promotional gate. Preserve actual
trajectory metrics, policy and the prior failed scientific gates unchanged.

## Interpretation Boundary

A fair future with y in {0,2} and constant p=c=1 has the exact Bayes conditional
mean yet a 50% empirical floor share. Thus a large floor alone cannot identify
architecture bias or justify violating E[H_E|x] <= E[H|x]. Singleton futures
do not separate conditional bias from irreducible future variation.

If substantial error remains inside the box, cap-only relaxation cannot
explain all failure. If the floor is large, compare fitting score-bin moments
and producer discrepancies before considering joint-cost training. Neither
case authorizes held-tuned shrinkage or opening reserved roles. A new training
hypothesis must retain the valid expected-cost ordering and strong controls.

## Execution And Resources

Register code/configuration and this protocol before pilot/run. Native arm64
environment, four CPU threads, interop one, workers zero. Use cached verified
scores, no new Torch inference required. One-view pilot estimates time/memory;
process per-view receipts with exact resume and a 10 GiB free-space reserve.
Check CREATE read-only; no job submission needed for this local diagnostic.

Obs8/pred12 native annotation steps and detector pixels. No metric/seconds,
true-3D, foundation, human-gold, physical-safety or publication-success claim.
Stage5C and SMC remain off. The full research goal is not achieved by this test.
