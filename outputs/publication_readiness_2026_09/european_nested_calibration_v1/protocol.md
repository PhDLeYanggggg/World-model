# Strictly Source-Separated Calibration and Support

## Question and Scope

Does calibration on sources excluded from the whole fitting chain, combined
with fitting-defined support and existing static guards, preserve a neural
advantage over equally protected damping? The previous loss-only experiment
failed the neural primary contrast, while improving damping. Keep both losses
as fixed controls, not held-result-selected winners. This is a new registered
development experiment on the same twelve opened source-training localities.
Independent selection/calibration/confirmation roles stay closed. No new
forecast model, score-head training, tolerance relaxation or deployment change.

## Frozen Producers and Source Roles

Reuse all nine frozen producer/seed forecast banks. Each producer fits four
localities. Each of the two other four-locality rosters acts as controller.
Utility/easy heads and their normalizers fit only that controller roster. For
each risk objective, equally average the four existing three-source heads;
their fitting union is the controller roster, never the outside roster.
Do not pick a component head or objective from its held results.

The remaining four localities were excluded from the forecaster, all heads and
normalizers. Enumerate all six two-calibration/two-evaluation splits within that
roster, fixed by sorted identities. These are dependent development views, not
six independent studies. Both calibration and evaluation localities are absent
from the entire fitting chain. Outcomes of one view may have been seen in other
development views historically; no independent-confirmation claim follows.

Before reading calibration outcomes, commit the protocol, then compute/freeze
all causal scores. Each row retains past-only features, frozen candidate and CV
rollouts, utility/easy moments, last-step motion and source-fitted support.
Mean four-head signed score is divided by the frozen utility head's fitting-only
CV cost scale. Do not interpret signed-objective components as moments.

## Fixed Support and Policies

Support distance is RMS feature z-score using the existing utility-head fitting
mean/std. Its ceiling is the weighted 99th percentile over fitting rows, with
existing equal-locality weights. Unsupported rows fall back; there is no held
normalization, quantile fit or support threshold sweep.

All four policies retain last-step movement, predicted positive net utility and
predicted easy-harm/reference <=2%. Risk-score<=0 defines `guarded`. `supported`
adds the support ceiling. `calibrated` tightens the risk-score cutoff using only
the two calibration sources. `calibrated_supported` combines these factors.
Legacy full-four-source point decisions are an additional frozen reference;
the new four-component risk ensemble is not silently equated to that policy.

Threshold grid in units of fitting CV cost: -1,-.5,-.2,-.1,-.05,-.02,-.01,
-.005,-.002,-.001,-.0005,0, plus the all-fallback option. For each nonempty
threshold require EACH calibration locality: at least32 supported known selected
rows, selected positive harm/reference <=2%, zero reference-exact harm, net easy
degradation <=2%, and strictly positive all-ADE gain vs CV. Select the eligible
threshold with largest mean-locality net gain; ties prefer coverage then stricter
threshold. If none qualifies, fallback. Missing easy support is ineligible.
This is empirical selection, not a conformal or physical-safety guarantee.

Freeze every calibration choice and held action before reading the corresponding
held outcomes. Every implementation interface slices calibration labels before
passing them to the calibration function. Record failures and empty coverage.

## Outcomes

Primary: signed-objective `calibrated_supported` neural-vs-matched-damping ADE
gain on held sources, with paired locality bootstrap. Also report its change vs
`guarded` and `supported`, both objectives, coverage, all/easy/hard ADE gain vs CV,
FDE, positive-harm budgets, zero-reference absolute harm, tail error and worst
individual source/role/seed views. No retuning after held readout.

Average dependent role/seed/calibration views within each fixed locality, then
bootstrap the twelve locality means (3000 draws, seed71431). Three-seed breakdown
is mandatory. Empty selected-risk ratios remain undefined, not zero. Report
budget-violation counts separately so undefined ratios cannot hide abstention.
For the exploratory benefit screen require positive primary lower interval,
positive neural gain vs CV, easy degradation <=2% in every held view, no
reference-exact harm, and nonzero coverage in every locality. Report actual
selected-risk violations; passing mean benefit is not a risk certificate.

## Interpretation and Resources

Only two independent calibration localities per view cannot support strong
source-generalization guarantees. Learn then Test formulates calibration using
statistical tests; original Conformal Risk Control treats monotone losses.
Our ratio on the changing accepted subset need not be monotone. This finite-grid
empirical comparison does NOT implement either guarantee or multiple-testing
correction. References: [Learn then Test](https://arxiv.org/abs/2110.01052) and
[Conformal Risk Control](https://arxiv.org/abs/2208.02814). Two-source calibration
failure must not be repaired by reading independent roles or relaxing 2%.

Run native arm64 CPU4/interop1, workers0, no hardware probing. New inference and
calibration only; checkpoint reuse is cached_verified, outputs fresh_run.
Record PID, timing, memory, resume and replay; preserve10GiB disk reserve.
Use CREATE only if measured resource need justifies it. No remote job changes.

Silver image-local obs8/pred12, raw-frame stride12: not seconds, metric,
human gold, true3D, foundation or certified physical safety. Not Stage37t50.
No Stage5C execution or SMC. Passing code tests is not a neural research success.
