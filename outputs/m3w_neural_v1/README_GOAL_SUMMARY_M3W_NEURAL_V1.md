# M3W Goal Summary: What Was Tried, What Failed, What Worked

Source status: `cached_verified` summary over Stage18-Stage41 evidence plus the fresh Stage41 completion and architecture audits.

This README is the consolidated "research ledger" for the M3W goal: train a stronger real-world multimodal multi-agent world model without overstating what the current evidence proves.

## One-Line Verdict

The project has moved beyond the original selector-only SDD result: the current best deployable candidate is **M3W-Neural v1 composite-tail safe-switch bounded neural dynamics under the Stage37/teacher safety floor**. It is positive on external dataset-local raw-frame benchmarks and has bootstrap, multiseed, pure-UCY, all-agent, endpoint-to-full, and ablation evidence.

It is still **not** a true 3D model, **not** a metric/seconds-level model, and **not** a large-scale foundation world model.

## Current Best Deployable Candidate

Current best deployable model:

```text
M3W-Neural v1 composite-tail safe-switch bounded neural dynamics candidate
safety floor = Stage37 selector / teacher floor
policy = composite_tail
switch_alpha = 1.0
tail_alpha = 0.08
calibrated_domains = ETH_UCY, TrajNet, UCY
uncalibrated_domain_rule = fallback_to_stage37_floor
Stage5C executed = false
SMC enabled = false
```

Main package metrics versus the Stage37/teacher floor:

| Metric | Value |
| --- | ---: |
| Gates | 41 / 41 |
| Rows | 55,528 |
| All ADE improvement | 0.2103 |
| t+50 ADE improvement | 0.1365 |
| t+100 raw-frame diagnostic ADE improvement | 0.1469 |
| Hard/failure ADE improvement | 0.2038 |
| Easy degradation | 0.0000 |
| Positive external domains | 3 |
| All-agent composite FDE improvement | 0.1982 |
| All-agent composite FDE@50 improvement | 0.1739 |

Bootstrap support for composite-tail:

| Slice | 2000-bootstrap low | Mid | High |
| --- | ---: | ---: | ---: |
| All | 0.2067 | 0.2102 | 0.2139 |
| t+50 | 0.1306 | 0.1366 | 0.1426 |
| t+100 raw-frame diagnostic | 0.1396 | 0.1469 | 0.1537 |
| Hard/failure | 0.1999 | 0.2039 | 0.2076 |

Multiseed support:

```text
composite_tail_multiseed_pass = true
all_mean = 0.2095
t50_mean = 0.1383
t100_raw_frame_mean = 0.1445
hard_mean = 0.2031
easy_max = 0.0
positive_domain_counts = [3, 3, 3]
```

Strict pure-UCY neural retrain/select/test evidence:

```text
gate = true
best_trial = pure_ucy_transformer
best_mode = bounded_endpoint_residual
all = 0.0901
t50 = 0.0880
t100 raw-frame diagnostic = 0.0831
hard/failure = 0.0936
easy = 0.0
bootstrap lows = all 0.0889 / t50 0.0863 / t100 0.0807 / hard 0.0923
```

Endpoint-to-full trajectory bridge evidence:

```text
two_domain_endpoint_to_full_gate = true
positive_domains = ETH_UCY, TrajNet
ETH_UCY bootstrap lows: ADE all 0.0150, ADE t50 0.0014, FDE all 0.0154, FDE t50 0.0020
TrajNet bootstrap lows: ADE all 0.0338, ADE t50 0.0186, FDE all 0.0339, FDE t50 0.0258
```

## Non-Claims That Must Stay Explicit

- This is not a true 3D world model.
- This is not a large-scale foundation world model.
- SDD remains pixel-space; external domains remain dataset-local / unverified weak-metric diagnostic.
- t+50 and t+100 are raw-frame horizons, not seconds-level claims.
- Homography, metric scale, and effective seconds remain unverified.
- Self-audited or visual-prior labels are not human gold.
- Stage5C latent generative rollout has not been executed.
- SMC has not been enabled.
- JEPA is representation-only in these experiments; it is not a generative world model.
- Ungated neural dynamics are not safe enough to deploy.

## Research Route Timeline

### 1. Early JEPA / WAM-Style Representation Attempts

Stages 18-19 tested SAM-JEPA-2.5D and WAM-style data strategy. JEPA did not collapse, but it did not improve selector, failure predictor, goal predictor, correction, or official t+50. The conclusion was that adding representation pretraining alone was not the bottleneck; the project needed stronger data, safer selector objectives, and external validation.

Result:

```text
JEPA non-collapse = yes
downstream lift = no
deployable contribution = no
```

Reason for failure:

- JEPA latents were not aligned with the actual deployment objective.
- The downstream heads needed causal history, hard/easy/failure structure, and conservative fallback more than generic representation variance.
- No latent rollout was allowed or useful at this point.

### 2. SDD Official Pixel-Space Benchmark and Selector Stabilization

Stages 22-26 built the SDD official pixel-space benchmark, SDD scene packs, lazy medium index, no-leakage audits, strongest causal baselines, hard/failure benches, and selector diagnostics. Stage26 became the SDD best deployable selector.

Key Stage26 status:

```text
t50 improvement ~= 14.58%
hard/failure improvement ~= 11.23%
easy degradation ~= 1.81%
Stage26 selector = current SDD best deployable at that time
```

What worked:

- Expected-FDE / cost-aware selector was better than hard baseline-class classification.
- Conservative fallback protected easy cases.
- Failure predictor and gain/harm logic were useful.

What failed before Stage26:

- Stage24 validation-selected hard-class selector had large oracle headroom but failed badly on deployment.
- It over-switched easy cases and produced negative t+50 improvement.
- Root cause: hard oracle labels were ambiguous, low-margin, and not cost-aware.

### 3. External Transfer and Domain Alignment

Stages 31-36 tried to move beyond SDD into OpenTraj / ETH-UCY / TrajNet / UCY style top-down pedestrian domains.

Initial zero-shot transfer failed:

```text
Stage31 zero-shot external all improvement ~= -92.67%
t50 ~= -278.57%
```

Reasons:

- Coordinate systems were incompatible.
- Scene/goal/interaction context was missing.
- Agent-type conventions did not match.
- Scale and homography were unverified.
- SDD selector was too SDD-specific.

Stage32-34 tried normalization, CORAL/latent alignment, coordinate-invariant features, relative-error targets, train-only external goals, row geometry, and scene packs. These produced local signals but not deployable all/easy behavior.

Stage35 improved all/hard but still failed t+50:

```text
all improvement = 12.13%
hard/failure = 13.98%
easy degradation = 0.041%
t50 improvement = 0.0
verdict = not deployable
```

Stage36 isolated t+50 as the blocker:

- t+50 had enough rows.
- t+50 had oracle headroom.
- Existing selector was not switching safely on long-horizon rows.
- Threshold tuning alone was not enough.

### 4. Stage37: External t+50 Repair

Stage37 introduced past-only history windows and scene-agnostic goal prototypes. This was the first external deployable positive transfer result.

Key result:

```text
all improvement = +13.48%
t50 improvement = +8.46%
t50 bootstrap CI = [+7.69%, +9.15%]
hard/failure improvement = +15.54%
easy degradation = 0.041%
gates = 16 / 16
verdict = stage37_t50_transfer_repaired_deployable
```

What worked:

- Past-only history windows: K=8/16/32/64 where available.
- Scene-agnostic goal prototypes: straight, stop, left/right turn, u-turn, group-follow, density-avoid, exit-like direction.
- t+50-specific switchability models: failure/gain/harm.
- Conformal or conservative safety guard.

Why it worked:

- It stopped treating t+50 as just another row in an all-horizon objective.
- It avoided held-out test-scene endpoint leakage by using relative motion prototypes instead of test goals.
- It switched only when gain was high, harm was low, and confidence was sufficient.

### 5. Stage38: Robustness and Bounded Correction

Stage38 froze Stage37 and tested cross-domain robustness plus bounded trajectory correction.

Outcome:

```text
Stage37 policy frozen = yes
external robustness = partial
bounded correction = not deployable
current best remained Stage37 selector
```

Why bounded correction failed:

- Raw correction could improve some hard/tail rows but did not safely dominate Stage37.
- Easy-case preservation was fragile.
- Correction did not yet provide enough stable dynamics lift to replace the selector floor.

### 6. Stage39-40: Neural Dynamics Attempts Before Breakthrough

Stage39 trained Causal Transformer, JEPA auxiliary, and Hybrid under Stage37 protection. Stage40 added teacher distillation, horizon heads, failure/gain/harm targets, hard/t50 curriculum, and automatic optimization.

Outcome:

```text
Transformer-only = trained but did not beat Stage37
JEPA = non-collapse historically, but downstream lift remained negative or absent
Hybrid = did not beat Stage37
Stage37 remained current external best
```

Failure taxonomy:

- Fallback consumption: the safety floor ate weak neural proposals.
- Ungated neural proposals harmed easy cases.
- JEPA representation did not align with deployment heads.
- Pure Transformer learned local dynamics but not enough safe switch/gain structure.
- Hybrid inherited JEPA's weak downstream contribution.
- t+100 remained especially hard without better long-horizon scene/domain context.

### 7. Stage41: Protected Neural Dynamics Breakthrough

Stage41 rebuilt the external split and datasets, then ran a large set of neural dynamics and safety-floor experiments.

Core data improvements:

- Fresh external split across ETH_UCY, TrajNet, and UCY-like domains.
- Past-only seq2seq world-model dataset.
- All-agent dataset with same-frame/past neighbor context.
- t10/t25/t50/t100 raw-frame labels for loss/eval only.
- No future endpoint input, no central velocity, no test endpoint goals.

Successful neural route:

```text
protected endpoint neural dynamics
teacher / Stage37 safety floor
self-gated endpoint candidate
composite-tail safe-switch bounded neural dynamics
endpoint-to-full bridge
all-agent composite world-state
```

Best same-protocol protected endpoint candidate:

```text
all = 0.4196
t50 = 0.4062
t100 raw-frame diagnostic = 0.4573
hard/failure = 0.4361
easy = 0.0
positive domains = 3
```

Frozen packaged deployable candidate:

```text
all = 0.2103
t50 = 0.1365
t100 raw-frame diagnostic = 0.1469
hard/failure = 0.2038
easy = 0.0
positive domains = 3
```

Why this route succeeded:

- It did not ask ungated neural dynamics to replace the floor everywhere.
- It used teacher/safety-floor logic to restrict where neural dynamics may intervene.
- It used bounded tail contribution rather than unbounded residual correction.
- It made all-agent and endpoint-to-full evidence explicit instead of only endpoint selector evidence.
- It kept test-time policy frozen and used bootstrap/multiseed evidence.

## Failed Routes and Why They Failed

| Route | Status | Main Failure Reason |
| --- | --- | --- |
| Stage18/19 JEPA-only representation | Failed for deployment | Non-collapse did not translate into selector/failure/goal/correction/t50 lift. |
| Stage24 hard-class selector | Failed | Oracle headroom existed, but low-margin hard labels caused over-switching and easy degradation. |
| SDD-to-external zero-shot | Failed | Coordinate, scale, scene/goal, agent-type, and horizon domain gaps were too large. |
| Raw normalization / latent alignment only | Failed | Reduced distribution gap did not produce predictive lift. |
| Mixed-domain selector without safety | Failed | Improved some averages but damaged easy cases, so not deployable. |
| Stage34/35 external global transfer | Partial / failed | Hard and t50 signals appeared, but all/easy/t50 were not simultaneously safe until Stage37. |
| Stage38 bounded correction | Failed for deployment | Could not safely beat Stage37 while preserving easy cases. |
| Stage39 pure Transformer | Failed | Negative or fallback-only; did not beat Stage37. |
| Stage39/41 JEPA-only | Failed | Stage41 JEPA auxiliary: all -0.0268, t50 -0.0151, hard -0.0243, easy degradation 0.0269. |
| Stage39/41 Hybrid JEPA+Transformer | Failed | Safe fallback-only or negative; no deployable dynamics lift. |
| Mixture selector | Failed | Safe fallback-only, no lift. |
| No-fallback neural | Failed safety | Often improved hard/all raw error but caused catastrophic easy degradation. |
| Continuous full-row bounded blend | Failed safety | all/t50/t100/hard positive, but easy degradation about 0.207, far above the <=2% gate. |
| Dynamic/calibrated/pairwise source switching after fixed composer | Mostly failed | Residual oracle headroom was tiny; truly positive residual rows were only about 0.1%. |
| Fixed-prior source switch | Failed | Did not beat fixed composer on both domains. |
| Learned full-waypoint shape alone | Weak | Safe positive contribution existed, but the learned shape gain was tiny and mostly tail-specific. |

## Successful Routes and What They Proved

| Route | Status | What It Proved |
| --- | --- | --- |
| Stage26 cost-aware SDD selector | Success on SDD | Expected-FDE, gain/harm, and fallback are necessary for robust selector deployment. |
| Stage37 causal history + goal prototype selector | Success external | External t+50 can be repaired with past-only history, scene-agnostic goal prototypes, and conservative switching. |
| Stage41 external split + no-leakage dataset | Success | Cross-domain external evaluation can be run with train/val/test discipline and no future/test leakage. |
| Stage41 self-gated endpoint candidate | Success as protected neural evidence | Neural endpoint dynamics can beat Stage37 under an internal safety gate. |
| Stage41 composite-tail safe-switch | Current packaged success | Bounded neural tail adds stable lift over teacher repair while preserving easy cases. |
| Strict pure-UCY neural retrain | Success | A source-heldout neural branch can produce positive bootstrap-stable lift on UCY-style data. |
| Endpoint-to-full trajectory bridge | Success | Endpoint neural dynamics remain positive when evaluated as full future waypoint rollouts on ETH_UCY and TrajNet. |
| All-agent composite world-state | Success | Evidence is not just single-agent endpoint selection; all-agent future world-state metrics are positive. |
| Ablation coverage | Success | No-history, no-neighbor, no-scene/goal, no-interaction, no-JEPA, no-Transformer, and no-fallback are all covered. |

## What The Model Actually Uses

The deployable path is not a plain JEPA or plain Transformer. It is a protected neural dynamics policy:

1. Stage37/teacher floor gives a safe baseline decision.
2. Neural endpoint dynamics propose bounded improvements.
3. Composite-tail policy allows switching or small tail contribution only under safety thresholds.
4. Easy rows fall back to the safety floor.
5. Unknown or uncalibrated domains fall back to Stage37.

Inputs remain causal:

- past history windows
- causal velocities and accelerations
- neighbor history / same-frame interaction context
- scene-agnostic goal prototype features
- horizon and domain metadata
- selected baseline rollout diagnostics computed without future labels

Future endpoints are used only as labels/evaluation targets, not inference inputs.

## Why This Is A World-State Candidate, Not A Foundation Model

It is a credible 2.5D multi-agent world-state candidate because it has:

- external top-down pedestrian domains
- all-agent future world-state evidence
- endpoint-to-full waypoint evidence
- hard/failure and easy-preservation gates
- bootstrap/multiseed support
- no-leakage audits
- safety fallback

It is not a foundation model because it lacks:

- large-scale broad video/image pretraining with proven downstream lift
- metric/3D calibration
- broad cross-dataset and cross-sensor generalization
- ungated neural dynamics safety
- foundation-scale model/data breadth

## Most Important Remaining Gaps

1. **Metric and time calibration**: verify FPS, annotation stride, homography, and meter-per-pixel before any metric or seconds-level claim.
2. **Ungated neural safety**: current neural success still depends on Stage37/teacher fallback; no-fallback neural is unsafe.
3. **JEPA contribution**: JEPA remains diagnostic-only; it has not produced deployable downstream lift.
4. **Long-horizon t+100**: now positive in protected raw-frame diagnostics, but still not seconds-level and still needs stronger long-horizon scene context.
5. **External breadth**: ETH_UCY, TrajNet, and UCY are positive, but this is still not broad foundation-scale validation.
6. **Full waypoint shape dynamics**: endpoint-to-full bridge works, learned shape contribution is positive but small.
7. **Publication package**: current evidence is strong for a candidate package, but claims must stay narrow: protected 2.5D neural world-state model, not true 3D/foundation.

## Direct Answers

```text
Did we train a neural world model? yes, protected neural dynamics were trained/evaluated.
Did neural exceed Stage37? yes, under Stage37/teacher safety floor and composite-tail policy.
Can neural replace Stage37 everywhere? no, ungated/no-fallback neural is not safe.
Is JEPA deployable? no.
Is Transformer useful? pure Transformer was not deployable; protected endpoint neural dynamics are useful.
Is current best still just selector-level? no, there is protected neural dynamics and all-agent world-state evidence, but it still depends on safety fallback.
Is this true 3D? no.
Is this foundation? no.
Can Stage5C run? no.
Can SMC run? no.
```

## Evidence File Index

- Main package: `outputs/m3w_neural_v1/report_m3w_neural_v1.md`
- Goal completion audit: `outputs/m3w_neural_v1/goal_completion_audit_m3w_neural_v1.md`
- Evidence matrix: `outputs/m3w_neural_v1/evidence_matrix_m3w_neural_v1.md`
- Architecture ablation: `outputs/m3w_neural_v1/neural_architecture_ablation_m3w_neural_v1.md`
- Ablation coverage: `outputs/m3w_neural_v1/ablation_coverage_m3w_neural_v1.md`
- Model card: `outputs/m3w_neural_v1/model_card_m3w_neural_v1.md`
- Data card: `outputs/m3w_neural_v1/data_card_m3w_neural_v1.md`
- Reproducibility: `outputs/m3w_neural_v1/reproducibility_m3w_neural_v1.md`
- Paper gap: `outputs/m3w_neural_v1/paper_gap_m3w_neural_v1.md`
- Frozen policy: `outputs/m3w_neural_v1/selector_policy_m3w_neural_v1.json`

## Recommended Next Work

1. Add new external top-down domains with legal scene images / trajectories, then rerun the same frozen-policy external protocol.
2. Complete metric/time calibration before any metric, seconds-level, or physical-world claim.
3. Train stronger full-waypoint/group dynamics, but keep Stage37/composite-tail fallback until no-fallback easy safety is proven.
4. Stop spending trials on residual source-switching around the fixed composer unless new causal scene/domain features are added.
5. Treat JEPA as research-only until it demonstrates downstream lift under the same external protocol.
