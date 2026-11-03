# Matched Unknown-Bound Cost Learning On SOURCE

## Material Passport

Prospective real Torch cost-head comparison. Frozen EqMotion forecasts and
SOURCE causal geometry/detector fields are cached_verified. New cost fits,
out-of-both-fit predictions and evaluation are fresh_run. This is exposed SOURCE
development,not a new final test or redefinition of the main-paper primary.

## Hypothesis And Scope

The fixed-support amplitude study rejects useful global attenuation alone.
The past-detector probe partially predicts wholly absent future labels,but is
not calibrated sufficiently for safety. Test the actual cost-learning mismatch:
known-only supervision omits the very unknown-row disagreement contributions
used by the existing evaluation bound. No probability from the probe is used
as a calibrated safety gate, and no future mask enters causal features.

Use the original149,510-window SOURCE controller whitelist,all74recordings and
twelve localities. Keep the three original disjoint SOURCE groups of four
localities. For each frozen forecaster trained on groupF,fit the critic on one
other groupG and evaluate on the remaining groupH. Rotate both G choices.
Thus evaluation sites occur in neither forecaster nor critic fitting. Three
actual seeds17/29/43 produce18critic jobs per arm,36new full fits overall.
Every locality has two excluded-producer views and three seeds,not extra
independent sites. No recording or difficult locality is removed.

## Matched Model And Training

Both arms use the existing width64 same-query-context cost head,444causal
fields,seven fixed rollouts,same initial weights,normalization,RMS,bias,source-
balanced cyclic query draws,AdamW,learning rate,clipping and complete budget.
The neural rollout is the stronger fixed EqMotion forecast; six physical
candidates remain for supervision and common preprocessing. The floor and
easy/hard cutoffs retain the original forecaster-fitting choices.

Common preprocessing uses only critic-fitting sites and ordinary observed
physical-candidate labels. New bounds do not change normalization or initial
outputs. The objective is the previous mean squared five-moment error plus
half absolute net-gain error,averaged within complete queries. Minimum2000
updates and four visits to every critic-fitting query; no slow-run reduction.
The only arm change is supervision support/semantics on unknown rows,including
the consequent query-average support normalization. Report fitting objectives
against both target definitions rather than comparing unlike losses as a win.

Known-only: retain observed masked-cost moments and omit wholly unknown rows
from supervised loss,while their causal rows remain query context.
Bound-aware: same observed moments; for wholly unknown rows use benefit lower0,
harm upperD,known-reference contribution0,known-easy-reference contribution0,
easy-harm upperD. D is causal candidate/floor maximum path disagreement.
Zero reference contribution is not an unknown zero trajectory error. Explicit
observed_future,target_defined and unknown_bound_supervised counters remain
separate. Partial futures retain the original available-label semantics.

Every checkpoint includes optimizer,sampler,RNG,draw hashes,normalizer,fit IDs
and PID. Perform50->100/direct100 recovery for both arms on each of the three
different critic-fitting groups,then complete first-job
refits for both. Freeze all36full fits and all causal cost/amplitude predictions
before reading held-SOURCE outcomes. Separate complete cost inference and metric
replay is required. Forecasts were already independently replayed; this study
does not mislabel another cache read as fresh EqMotion training/inference.

## Fixed Decision And Primary

Only the neural candidate intervenes in the primary decision,isolating target
repair from a changed physical-action selector. If predicted net benefit is
nonpositive,reference/easy-reference is nonpositive,or the rollout aliases the
floor,return alpha0. Otherwise choose the largest alpha<=1 satisfying both
predicted inequalities alpha*H<=.02*R and alpha*EH<=.02*ER. Prediction is
floor+alpha*(neural-floor). This uses convexity of fixed-mask ADE and the linear
unknown-disagreement bound. It guarantees only predicted checks,not true risk
or population safety. No outcome-selected confidence/alpha threshold is used.

Primary: bound-aware minus known-only bounded-independent paired completion-
lower gain,in percentage points of each original fixed known floor reference.
Known gain and paired upper remain secondary. Binary independent2%guard and
full EqMotion substitution are fixed secondary controls,not replacements for
a failed primary. No joint solver or pairwise conflict method is added here.
Those remain separate gaps for the broader method objective.

Report all/easy/hard gains,ADE,available-endpoint FDE,tail error,actual amplitude
distribution,active rate,predicted versus observed recording costs and full
original risk support. Empty selected reference remains not estimable,not a
safe zero. No epsilon micro-intervention is added to manufacture support.
Require the original2%all/easy positive-harm and signed easy-net limits,unknown
upper bounds and finite support in every required recording. Do not replace
the contract with a mean or a selected favorable recording subset.

Use3000locality bootstrap resamples after averaging the two excluded folds and
actual seeds within each of twelve SOURCE sites. No threshold/model/seed is
selected from these scores. This changes no official final-evaluation role or
metric. Available-label ADE is not complete12ADE; bounds cover wholly unknown
rows,not missing steps inside partial futures. No selection/development,private
calibration,final confirmation or DroneCrowd arrays are opened.

Local-first native arm64 Torch CPU4/inter-op1/workers0,atomic checkpoint,
heartbeat,unchanged resume,10GiBdisk reserve and12hphase boundary. Real pilot
estimates the full budget. No fallback is presented as Torch training.
No deployment promotion,pixel-to-metric/seconds/human-gold claim,true3D or
foundation claim. Stage5C and SMC remain off; no readiness follows from code
tests or an observed SOURCE component gain.
