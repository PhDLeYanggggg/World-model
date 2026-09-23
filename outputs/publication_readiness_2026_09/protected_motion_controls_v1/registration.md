# Matched Protected Motion Controls

Registered before new fitting or outcome readout, 2026-09-24.

## Question

Does a learned full-trajectory forecast add value beyond applying the same
benefit/harm learner and conservative intervention rule to simple causal motion?
The DUT descriptive readout showed that comparing a protected neural forecaster
with *unprotected* damping cannot answer this question. No DUT-driven choice of
damping rate is made: all six non-CV actions in the existing seven-baseline family
are included. This is a falsification experiment, not a new deployment policy.

## Fixed Design

- Four development-exposed SDD physical sites; 33 recordings; all 175,756
  past-eligible rows; eight observed and twelve predicted native annotation
  steps. Source stride12 is not a verified time unit.
- Outer leave-one-site exclusion and seeds17/29/43. The neural forecast's
  training-cost labels come from pair-excluded producers: neither the outer site
  nor that label row's site enters the corresponding forecasting producer.
  Causal actions need no trained producer. Cost heads and preprocessing use
  only the three other sites, with complete future paths as training supervision.
- Six actions: position, damping005/010/020, acceleration, turn rate. CV is the
  reference/fallback, not a seventh switching action. Full Transformer is the
  matched learned-forecast comparator, not a ramp surrogate.
- Identical 356-feature causal cost interface, bounded-fraction benefit/harm
  objective, 3,000 updates, batch256, width64, AdamW0.001. Reuse the twelve
  existing full-Transformer bounded-fraction heads only after exact hash,
  feature/label/preprocessing/support/budget checks. Fit72 new causal heads.
- Fit84 forests (seven actions x twelve folds/seeds), 128 trees, depth16,
  leaf64, max_features1/3. Tree weights equal the paired neural sampler counts;
  zero-disagreement rows have zero optimization mass in both objectives.
  Optimizer budgets are not claimed equivalent across neural/tree families.
- No hyperparameter/threshold search, early stopping, held-out model selection,
  DUT prediction or DroneCrowd readout. No calibration certificate. All fits
  complete before new outer decisions; all decisions saved before evaluation
  labels load. Inference membership never depends on future-label availability.
- Primary rule: moving and nonzero candidate-CV disagreement, predicted benefit
  greater than harm, and predicted harm <=0.1*benefit. Net-positive-only and
  uncontrolled predictions remain visible diagnostic controls. Easy/hard cutoffs
  reuse the frozen parent training-only q25-positive-CV/q75-CV values, unchanged
  across candidate actions.

## Estimands and Failure Criteria

Primary descriptive contrasts: full Transformer minus **each** protected causal
action under the strict rule, separately for neural and forest cost learners.
Report all contrasts, without choosing a winner for deployment. Secondary:
neural-vs-forest heads within each action. Also compare each causal action with
Transformer at matched strict intervention count in each fold/seed: use the
smaller strict count, retain each arm's highest predicted net gains inside its
own strict eligible set, break ties by global row ID. This is an outcome-blind
offline budget diagnostic, not a per-query deployable rule.

Primary metric remains the existing mean of four scene-relative available-point
ADE gains over CV, not pooled-window gain or a new DUT-style estimand. Report
FDE, complete paths, positive-easy, hard, zero-CV harm, per-scene/seed counts,
tail errors, missing-label full-grid gain bounds, and paired physical-site
bootstrap (3,000; four sites only). No overlapping-window IID bootstrap. Report
seed spread separately; mean-of-seed errors is not a forecast ensemble.

If a simple protected action matches or dominates neural forecasting, or the
paired uncertainty does not support an incremental neural gain, retain that
negative evidence. Easy mean preservation is not individual safety. Already
development-exposed source folds cannot become independent confirmation.

## Scope and Runtime

EqMotion has prior fixed DUT evidence but is not silently added to this source
contrast without verified pair-excluded training-cost producers. A new nested
EqMotion bank remains separate work. This experiment establishes neither a
joint-interaction contribution nor a multimodal contribution.

Native arm64 Python, CPU4/interop1, workers0. Atomic checkpoints and heartbeats,
explicit resume, one runner lock. First fixed fit is a runtime pilot and continues
unchanged to the complete matrix; slowness alone is not a stopping criterion.
Weights, row-level decisions and caches stay local. Reports, configuration, code
and lightweight metrics may be committed. Stage5C and SMC remain disabled.
