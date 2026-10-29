# Causal Reference Coverage and Action-Bank Support

29 October 2026. Fresh neural training and diagnostic computation; frozen
assets and controls are cached_verified. **No new deployment or submission-ready
claim. The full M3W research goal remains incomplete.**

## 1. Controlled Neural Experiment

The preceding harm-aware loss comparison exposed a geometric restriction:
some CV trajectories were outside the bounded neural output set. This run
changes only the radius:

```
B_new = max(B_old, 1.05*distance(CV,center), 1.05*distance(D97,center)).
```

Both references and the radius use observed history only. Architecture,
parameter count, input fields, source folds, seeds, loss2+2, normalizers,
sampling, initialization, AdamW and4,000updates remain fixed. The margin was
not searched. The current producer is a grouped agent-track Transformer, not
a new JEPA/hybrid or generative model.

All9fits/36,000primaryupdates complete: PID43521,exit0,537.401863seconds,
peakRSS1,580,204,032bytes. The80update real pilot verifies exact interrupted
resume and coverage-off/control equivalence. A full4,000step neutral replay
on the affected fold is exact. Six unchanged-radius fits reproduce their
complete preceding model/optimizer/RNG/draw state and forecasts. Three
affected fits retain exact sampling/RNG states; parameters remain88,514.

All endpoints were committed in8fc3f29a before scoring. There was no
checkpoint/seed/threshold selection. Initial training proposals, scope
implication and all earlier failures remain preserved.

## 2. A Small Predictive Repair, Not a Stronger Safe Model

Equal-locality gains over causal CV, averaged over repeated producer/seed
errors within locality:

| Endpoint | Harm-aware control (%) | Reference coverage (%) | Paired change (pp) |
|---|---:|---:|---:|
| All ADE | 7.531916 | 7.595475 | +0.063559 |
| Complete-future ADE | 11.053453 | 11.130813 | +0.077360 |
| Easy ADE | -8.433059 | -8.345904 | +0.087155 |
| Hard ADE | 11.675472 | 11.734517 | +0.059045 |
| All FDE | 11.588928 | 11.666115 | +0.077187 |

The primary conditional95%paired locality-bootstrap interval is
[+0.029923,+0.096635]pp,using3,000draws over12localities. Seed differences
are+0.078978,+0.060200,+0.051499pp. Seven localities improve,one regresses,
four are unchanged. This is a small coherent source-development effect,
not independent confirmation or a meaningful world-model success by itself.

The original ADE-only producer remains better overall: its gain is8.461640%,
and the new-minus-original contrast is-0.866165pp,CI[-1.582816,-0.109369].
Easy degradation remains8.345904%,above2%. Positive harm is separate from
net error: all-row harm versus CV changes11.756180%to11.744020%,while harm
versus the train-selected reference slightly increases2.472366%to2.481350%.
Do not summarize this as every risk measure improving.

Known unrestricted CV-harm failures remain963/978all and922/978easy
recording/producer/seed views. These repeat163recordings,not978independent
ones. Every all-row view contains some unknown future labels. This is
candidate forecasting,not a newly trained/calibrated full intervention policy.

## 3. Original Fixed-Candidate Support Still Fails

All148required TRAIN recording/context views are retained. Original,harm-aware
and reference-coverage candidates give146,147and147feasible/positive views.
Both preceding controls replay exactly. Coverage gains0and loses0versus the
harm-aware control.

The remaining view is `single1_seed43_controller2_dimensionless`,locality082,
recording51. It has78fitting rows and3known eligible easy rows. None of those
three neural actions is within the2%ratio; the minimum is1.931046,not0.02.
This producer's radius and final state are unchanged,so that result was
predicted by the pre-readout scope note. No recording is removed.

## 4. A More Useful Next Direction: Real Alternative Actions

A label-aware case probe finds genuine damping/OLS alternatives on the
remaining recording. Exact analytic equality to the floor is removed rather
than counted as a fake safe intervention. This motivated a prospectively
registered full-roster action-bank probe,committed in86705f11 before its
readout. It uses the same cached TRAIN packets,floor,eligibility,labels,
reference/event definitions and2%rules,with at most one real action per row.

| Fixed bank | Label-aware budget/positive feasibility |
|---|---:|
| Original neural candidate alone | 146/148 |
| Reference-coverage candidate alone | 147/148 |
| Six existing causal physical baselines | 148/148 |
| Physical bank plus original neural candidate | 148/148 |
| Physical bank plus coverage candidate | 148/148 |

The physical bank is position,CV,damping.90,damping.97,OLS4andOLS8. No-op
choices are excluded. These are **oracle feasibility counts,not passed learned
policy gates**. They do not replace the146/147historical failures. No action
is selected by future labels at inference,because no new inference policy is
created by this diagnostic.

The original neural action adds positive oracle utility over physical-only
choices in all148views. A descriptive aggregation gives2.743738percentage
points of full floor-reference cost: within each source/context,sum the
neural-bank minus physical-bank lower-utility masses across recordings,divide
by the full known reference mass counted once per row,average the two contexts
within each locality,then average12localities. This is an oracle potential,
**not learned ADE/FDE lift or an independent confidence interval**. It uses the
fixed seed43producer contexts,not a selected best seed. Per-record relative
increments range0.296393to11.793123pp,median2.170115pp.

The result changes the next action. More neural output capacity is not the
only missing ingredient: the old binary action set discards real useful
alternatives. A prospective multi-action gain/harm learner should be tested
against a physical-only learner,static TRAIN-selected motion and the existing
binary neural policy. Neural availability must demonstrate a learned gain
over the physical-only control; a favorable oracle is insufficient.

## 5. Verification and Recovery

- Predictive readout PID44501 exits0after288.009012s. ReplayPID44878 exits0
  after32.762817s; JSON and CSV are exact,with2,304comparator metric checks.
- The first support attempt PID44928 exits1after a read-only SSH255. No job,
  authentication or training action is taken. The underlying connection cause
  is not established; a later identical-asset read succeeds.
- A private hash/byte-verified cache permits recovery. PID45598 exits0after
  69.668609s,all148views complete. PID46125 exits0after18.819719s,exact replay
  from cached bytes with no network. The24packets occupy about420MiB and are
  excluded from Git; cap4GiB,reserve10GiB.
- The first remaining-case probe is rejected before any report because it
  used a mean instead of maximum disagreement for partial-label bounds. The
  corrected probe calls the original max utility and has a regression test.
  Its final result replays exactly. Original results are not changed.
- Full-bank PID48722 and replayPID49327 both exit0with identical result hash.
  All444single-action controls reproduce; the bank oracle passes80exhaustive
  small synthetic cases. Synthetic checks are not real-world success.
- 70 targeted tests pass in 3.45 s. The full historical suite was not rerun.
- Repeatable verification wrappers retain the first execution receipts while
  recomputing the frozen numerical checks. Real readout/support repeats complete
  with exit 0; changed scientific hashes still fail. The registered numerical
  routines and old scientific results are not edited to accommodate runtime variance.

## 6. Scope and Remaining Work

There are318,969source windows,163recordings and12already-opened localities;
7,047rows have no future labels and193,705have complete requested futures.
Eight observations/twelve predictions,stride12raw frames,is not raw-frame
t+50. Data are image pixels and released detector tracks,not human gold.
The model uses local trajectory/neighbor context,not a new scene-image,
JEPA,occupancy or physical-world contribution.

Independent selection,calibration and confirmation remain unopened. No new
multi-action selector or candidate-dependent cost head is trained here.
Matched-count controls remain utility-only; empty risk is not_estimable.
The original2%full-policy requirements remain on every required recording.
Deployment,Stage5C andSMC remain unchanged/off. No metric,seconds,true3D,
foundation or A-conference readiness claim is supported.

The highest-value next experiment is a bounded **learned multi-action
baseline-relative policy**,using the stronger original frozen forecaster and
legal physical alternatives,with physical-only and old binary controls.
Keep the existing role boundaries,three training seeds and fixed primary
utility/safety readout. Only after a useful fitting/validation result should
independent calibration/confirmation be opened under their existing contract.
That training is **not_run**. This iteration is finished; the full goal is not.
