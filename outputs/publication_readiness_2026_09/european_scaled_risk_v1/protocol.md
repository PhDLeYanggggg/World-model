# TRAIN Budget Scaling: One-Factor Matched Repair

## Material Passport and Hypothesis

Experiment run, following the complete TRAIN-only gradient diagnostic. All 216
original fixed monitors replay. In the task-risk arm, median weighted easy-risk
gradient/data-gradient norm is 0.000698, while 61/72 soft and 52/72 hard TRAIN
monitors have positive easy excess. The median easy reference budget is 0.00041855
in normalized cost units. These are descriptive mechanisms, not a proved cause.

Hypothesis: positive TRAIN-reference scaling improves empirical constraint
optimization and subsequently selected risk without destroying decision utility.
Failure remains possible. We change only constraint units in primal and dual
updates, not model capacity, features, decision thresholds or evaluation rules.

## Fixed Intervention

For c_all = mean_query[a*(H-.02R)] and c_easy = mean_query[a*(EH-.02ER)],
compute s_all=.02*mean_query[R] and s_easy=.02*mean_query[ER] using all known
TRAIN queries, in the existing TRAIN normalization. Average agents within query,
queries within locality, then localities equally. No monitor or evaluation labels
determine these constants. If either is zero, stop as unsupported rather than
invent an epsilon-supported constraint.

Unit control uses scales [1,1]. Candidate divides each constraint by its positive
TRAIN scale in both the primal loss and projected dual update. Positive scaling
preserves the constraint zero set, but changes optimization and effective dual
capacity. Dual learning rate 1, cap 100, temperature 1 to 0.1, primary/utility
loss, initialization, sampler, width 32 and 2,000 updates remain fixed. Report
caps and original-unit violations; a smaller normalized loss is not a pass.

Use the frozen original trainer inside a single-process scoped objective adapter,
restoring functions on every exit. No registered source is edited. The checkpoint
identity binds mode and TRAIN scales. Unit control must reproduce the previous
task-risk checkpoint exactly for all 72 identities, not only the pilot. Train
both arms fresh: 144 fits. Earlier quadratic and utility controls are explicitly
cached_verified, not falsely labeled fresh; their matched lineage remains fixed.

## Execution and Engineering Checks

Native arm64 CPU, four compute threads, zero workers, streamed hash-verified
TRAIN packets. Preserve 10GiB reserve and 256MiB new checkpoint cap. Pilot the
first identity for 100 updates per arm, including interrupted candidate/direct
replay and original control: 400 updates. Check runtime, RSS, finite loss, exact
resume and unit-control equality. Same 12-hour limit, atomic checkpoints,
heartbeat and explicit resume. Neither pilot loss nor monitor risk selects a
model. Freeze all 144 final fits before any new development inference.

## Fixed Evaluation and Advancement

Compare budget-scaled with fresh unit control and cached-verified quadratic,
task-utility and original forest. Preserve additive, Poisson and cost forest
secondary controls. Use full and same-query matched-count paired lower utility,
all moments, original hard actions, positive selected easy harm/reference error,
completion support, unknown outcomes, intervention and per-locality results.

Require positive nominal 95% lower utility intervals for all eight primary
contrasts (four controls, full and same-count) and complete finite support with
no 2% easy-risk violation for candidate full and primary matched policies.
Use 3,000 paired-locality bootstrap draws over 12 localities, averaging repeated
views first. Empty or unsupported selections do not pass. Do not choose a new
threshold or checkpoint from this readout. These are exposed development results,
not independent calibration or confirmation. A failed repair leaves deployment
unchanged; it is not rescued by a weaker comparator or fewer selected cases.

## Prior Art and Limits

Scaling constraints is numerical conditioning, not a new safety theorem. The
proxy-Lagrangian literature explicitly distinguishes surrogate optimization from
enforcement of original non-differentiable constraints. Our deterministic fixed
head/dual update is not their stochastic classifier algorithm or its guarantee.
[Cotter, Jiang and Sridharan, ALT 2019](https://proceedings.mlr.press/v98/cotter19a.html).
This experiment does not yet repair soft/hard mismatch; retaining it isolates
the scaling factor. Observation8/prediction12 and dataset-local units are unchanged.
No seconds, metric, true3D, foundation, Stage5C or SMC claim or execution is added.
