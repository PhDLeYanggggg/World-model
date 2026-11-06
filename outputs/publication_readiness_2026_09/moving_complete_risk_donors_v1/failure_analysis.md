# Complete-Risk Supervision Does Not Repair The Policy

## What Was Actually Changed

24 new Torch risk heads, risk seeds 17/29/43, and 24 complete exact refits.
Each fit uses 4,096 updates. The intervention restores 16,944 unknown producer-A
candidate rows and 1,274 unknown own-B proposals omitted by known-only fitting.
Known features and labels are exact. The network, 120-dimensional causal basis,
proposer, forecast teacher, zero threshold and original risk evaluator stay fixed.
Source/query reweighting and fitting-only normalization change with the complete
pool; they are part of the treatment, not separately identified causes.

## Result: Not A Safety Repair

Mean known-label source-macro gain falls from 1.313033% to 1.134321%; unknown
admissions fall from 81 to 68 but total interventions rise from 2,730 to 3,359.
Known-easy harm violations increase from 15 to 19. Only 1/24 recording views
passes the original contract, unchanged from before. Unanimity falls from 2/24
to 1/24 supported views, with 14 undefined easy-reference views instead of 12.
All three individual risk-seed policies pass 0/24 views.

The mean's separate completion lower gain rises from 0.823788% to 0.921428%,
but this is not demonstrated paired improvement. The paired completion-lower
contrast is -0.783298 pp, exploratory interval [-2.085325, +0.095371]. The two
policies have different unknown paths; the paired bound is on their difference,
not the difference between two separately pessimistic floor-relative bounds.
The unanimity paired lower is -0.154287 pp [-0.441157, +0.120787]. No new policy
passes. Do not promote the favorable separate lower gain in isolation.

## Failure Taxonomy

### 1. Bound Regression Competes With Observed-Risk Learning

On the fixed known-only fitting-RMS scale, observed-target source/seed-macro
MSE worsens from 1.715678 to 23.081753, with 0/24 improved source/seed views.
Bound-target MSE improves from 586.586965 to 527.056177, with 21/24 improved
views. The latter is error against an envelope contribution, not unobserved
ground truth. Restoring unknown supervision is therefore not sufficient:
the shared regression trades away useful observed-risk prediction.

### 2. Training Scales And Pool Composition Move

Training-target RMS rises by median factors 2.333 for all risk and 5.917 for
easy risk, up to 6.368 and 10.024. Conservative upper contributions have much
larger magnitude than many observed signed excesses. This is consistent with
bound targets and changed normalization disrupting the loss balance; it does
not isolate normalization from changed sampling/targets. A fixed-scale or
factorized-target ablation is needed before assigning a single causal mechanism.

### 3. Pointwise Excess Veto Does Not Ensure Reference Support

Two predicted signed excesses do not model the selected easy-reference mass or
enforce the recording-level risk ratio. Fewer unknown admissions can coexist
with more known harm and missing support. Unanimity still has zero signed
easy-net violations but fails positive harm/support. Benefit cannot cancel
positive harm under the original contract. Threshold tightening or empty-output
certification is not a repair under the user's retained rules.

### 4. Observability And Generalization Remain Unresolved

The unchanged basis contains motion, relative-neighbor and rollout summaries,
not a dedicated visibility/annotation-observability model. Some censoring may
not be identifiable from those summaries. This is a feature hypothesis, not
evidence that every missing label is a scene exit. Only eight exposed sources
and one forecaster seed are involved; new independent confirmation is absent.

## Next Model-Side Test

Separate observed signed-risk learning from the unknown-envelope process.
Retain the known-only risk head as a frozen control; learn past-only
observability/envelope contributions with separate normalization and losses,
and explicitly model reference/support moments. Compare this against the
current shared regression at the same source exclusions and budget. Any
visibility features must be lawful past/current camera context, not actual
future-validity masks. Evaluate complete policies with the same original rules.

No new C/H, identity exemption, denominator change, threshold search or hidden
test tuning. No ordinary residual, Stage5C or SMC. The original 24-group main
remains NA. These are native obs8/pred12 image-coordinate silver-label
development results, not true 3D, metric/seconds, foundation or submission success.
