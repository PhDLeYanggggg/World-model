# Nested Gain/Harm and Joint Intervention: Safety Not Repaired

## Completed Experiment

`fresh_run`: nine ridge cost heads and nine real Torch cost heads, three seeds
and three outer source folds. Each neural head reaches exactly 2,000 updates
(18,000 total); ridge is a closed-form fit, not neural training. The predictors,
head matrix, cost loss and control parameters were specified before comparative
outcomes were used. A receipt-path typo was repaired and regression-tested
before the first cost-head launch; no loss, threshold, budget or split changed.

The outer fold is excluded from preprocessing, both upstream cost-label
producers, the cost head and the final predictor. For outer A, the head learns
on B using C-only predictions and on C using B-only predictions; its candidate
on A comes from a B+C predictor. This avoids outer-source label leakage, but
creates a four-site-versus-eight-site producer-size shift. That shift remains a
limitation rather than an experimentally isolated explanation of failure.

Targets are continuous positive ADE benefit and harm, not best-class labels.
Features use causal geometry and candidate/fallback rollout disagreement.
The neural loss weights harm underestimation fourfold. Its harm output is an
upper-expectile-type score, not a calibrated conditional mean or upper bound.

## Results

Full-cohort pointwise learned gating has positive average gain over the selected
floor: ridge 3.5638--4.0131%, neural 3.2851--3.5873%. But positive-easy degradation
still ranges from 12.8069% to 14.3825% across those rules/seeds. Some rules still
harm exact-zero-CV cases. The chosen floor itself fails CV-relative easy
preservation, so even a zero-switch rule does not solve that requirement.

Joint controls use a separate fixed past-selected pilot: **1,152 queries and
6,116 target rows**. It is not the 318,969-target full forecast cohort. All joint
controls share the same pilot, including targets whose outcomes are unknown.

| Neural cost-head seed | Joint ADE gain vs floor | Conditional locality CI | Easy degradation vs CV |
|---|---:|---|---:|
| 17 | 0.3070% | [0.0223%, 0.5993%] | 10.9033% |
| 29 | 0.4842% | [0.2255%, 0.7811%] | 10.8084% |
| 43 | 0.5261% | [0.1231%, 1.0331%] | 11.1035% |

These small gains do not establish a joint-decision contribution. At exactly
the same nonzero intervention count per query, joint versus independent ADE
gain is **0.0000% for all three neural cost-head seeds**. Joint versus the unary
geometry control is -0.000192%, 0%, 0%. Ridge exact-count joint versus independent
is +0.001776%, -0.108901%, -0.098659%. In the last case its conditional interval
is entirely negative. Retain those controls; do not report only joint versus
the weaker floor. Full tables and paired intervals are in
[results](results.md) and [matched contrasts](paired_joint_contrasts.md).

All recorded solver controls complete without an optimality-failure flag.
Actual binary decisions were independently rechecked for equal query-level
intervention counts. Solver feasibility is not forecasting or physical safety.
The pair term is a proximity proxy on predicted image-plane trajectories, not
observed collisions or verified ground-plane interaction risk.

## Why This Version Is Not Sufficient

1. **Reference mismatch:** the requested easy constraint is CV-relative, whereas
   the policy budgets predicted harm relative to a training-selected floor that
   already damages easy cases. This is a demonstrated contract mismatch.
2. **Prediction is not calibration:** a 2%-of-training-error predicted harm cap
   does not bound conditional actual easy harm. No independent risk calibration
   was opened or passed.
3. **Weak/nonuniform risk ranking:** seed17 neural signed-gain correlation is
   negative at localities112 and048 and near zero at082. This is descriptive
   evidence of locality heterogeneity, not proof of a single causal mechanism.
4. **Joint term has no stable predictive value:** same-count controls are null
   or unfavorable. Reduced proxy overlap alone is not a trajectory gain claim.
5. **Producer shift and sparse exact-zero support:** both need a new controlled
   experiment. Neither is repaired by calling a conservative score conformal.

## Verification and Decision

`cached_verified`: all cost/prediction/decision hashes are checked and the
complete readout is recomputed without new solvers. `fresh_run`: checkpoint
inference on 4,096 excluded rows per head reproduces all 18 saved score subsets
exactly. All matched query counts pass a separate decision-array check. There
are 127 scoped regression tests, not a complete repository-suite claim.

No deployment promotion. No validated joint mechanism or statistical safety
certificate. Reserved selection/calibration/confirmation data remain closed.
No metric, seconds, true-3D, foundation or paper-readiness claim. Stage5C/SMC off.
Future repair should separate a CV-safe reference contract from average-utility
model selection and isolate risk prediction before spending more complexity
on joint geometry. No new threshold is selected from this completed readout.

[Analysis](analysis.json), [verification](verification.json),
[checkpoint replay](checkpoint_replay.json), [training losses](training_losses.md),
[method and assumptions](method_draft.md),
[execution record](../european_source_execution_20260924.md).
