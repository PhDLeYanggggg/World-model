# Matched Two-Sided Budget-Boundary Cost Learning

## Question And Prior Controls

Can direct supervision of both sides of the unchanged 2% decision boundary
improve a frozen published-loss ranker's complete guarded policy, beyond an
equally trained coherent moment critic? This is a SOURCE mechanism experiment,
not an independent test or a new deferral theorem.

Previous symmetric signed-risk MSE, 4x underprediction weighting, completion-
endpoint proxies, reference coherence and own-choice query regression remain
completed negative controls. The new factor is boundary-event log loss tied
directly to the same cost outputs used by the guard, rather than a separately
calibrated confidence or best-expert classifier. It can repair errors in either
direction. That does not guarantee improvement or make BCE a new method.

## Fixed Factors

Both arms start from the same completed own-choice critic for each context/seed,
with fresh AdamW. Each retains the same 31,119 active weights, coherent decoder,
462 causal fields, initial selected-set context, source-balanced complete cyclic
query draws and fixed coverage budget. Both use the original fixed-initial-choice
query residual loss during this continuation; neither runs integer allocation in
the training gradient. Thus this is not another own-vs-fixed-choice comparison.
Two exact first-context full-budget refits and a 400-update interruption/recovery
pilot are required. All 36 fits and 72 readout groups are mandatory.

The final current-bank TDEF scores and all seven forecast paths are frozen.
Only the two newly trained critics' original pointwise guards differ. No
temperature, threshold, candidate, reference denominator or risk tolerance is
selected from outcomes. No new trajectory or independent role is opened.

## Objective

The moment arm retains the preceding pointwise B/H/R/ER/EH and query-residual
objective. The boundary arm adds a unit-weight auxiliary:

```text
z_all  = log(Hhat + eps) - log(0.02 * stopgrad(Rhat) + eps)
z_easy = log(EHhat + eps) - log(0.02 * stopgrad(ERhat) + eps)
l_all  = 1[H > 0.02 R]
l_easy = 1[EH > 0.02 ER]
auxiliary = mean BCEWithLogits(z, l)
```

Costs are in the same frozen causal normalized units. Eps=1e-6 is a numerical
log safeguard, not a native risk denominator or a deployment tolerance; the
unchanged evaluator never adds it to risk ratios. Averaging is over known legal
actions per row, known rows per query, then the full 16-query batch. Unknown
rows remain wholly NaN labels and have no auxiliary or observed-cost loss.
They still participate in the causal query context and draw pool.

The auxiliary's reference is detached to prevent a direct denominator-inflation
gradient. Original reference regression remains trainable. The shared encoder
can still indirectly change references, so this is not an exact reference lock.
Boundary targets are realized loss labels, not inference features. The resulting
cost outputs are risk-oriented proxies; their interpretation as conditional
means or calibrated event probabilities is not guaranteed. Report the tradeoff
with native cost errors and true policy risk rather than treating BCE as safety.

## Readout And Advancement

Required policies: moment-guarded TDEF, boundary-guarded TDEF, frozen original
guarded TDEF, frozen own-joint, raw TDEF, and the floor. Retain the original
causal availability, positive B-H, positive R/ER, H<=0.02R, EH<=0.02ER rules.
Future labels/validity/easy status never enter this filter.

Primary: boundary minus moment FULL paired completion-lower utility, nominal
3000-locality bootstrap 95% lower endpoint positive. Also require positive lower
effect versus the frozen guarded TDEF, positive lower utility versus the floor,
and complete original all/easy recording risk with easy preservation. Report
all original 149,510 targets, 74 recordings, 12 localities and 444 repeated risk
views, not a favorable subset. Count matching is utility only; a required empty
view is not estimable, not safe zero. Missing outcomes are uncertainty, not
observed harm. Cost seeds17/29/43 share frozen forecaster43.

The registered secondary diagnostic reconstructs false admissions, false
rejections and easy-reference support for both new guards only after choices
are frozen. These are realized-label counts, not proof of conditional calibration
or independent samples. Raw means or a repaired support count cannot rescue a
failed primary or safety gate. No deployment is promoted from SOURCE fitting.

## Runtime And Role Boundaries

Use native arm64 `.venv-pytorch`, CPU4/inter-op1/workers0, no multiprocessing
or resource probing. Atomic checkpoints every50 updates retain optimizer,
sampler, random state and objective identity; heartbeat every50. Explicit
unchanged-definition resume only. Preserve10GiB disk and1GiB private checkpoints;
12h per-phase limit is resumable, not a reason to shrink samples/budgets.
Measure real pilot time/RSS before committing the full run. Use CREATE only if
the measured scale requires it; check existing jobs before any submission.

SOURCE remains exposed detector-silver/oblique local-image obs8/pred12/stride12
raw-frame development, not SDD t+50, independent confirmation, human gold,
metric/seconds, physical safety, true3D or foundation evidence. Stage5C and SMC
stay off. No reserved DroneCrowd/selection/calibration/confirmation role is used.
