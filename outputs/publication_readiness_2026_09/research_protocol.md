# CVPR 2027 Research Protocol Draft

Status: proposed protocol, not approved. The source/teacher audit found historical contamination; scientific choices and independent calibration/confirmation assets remain unresolved. Target venue selected by the project owner on 2026-09-16: CVPR 2027. This is not a preregistration or a completed experiment.

This document retains the **independent-confirmation draft**, not the current
development run. The user subsequently selected obs8/pred12 with raw-frame
t+50 supplemental and delegated the research route. Development was approved
under `configs/m3w_8to12_continuous_context_v5.json` and the subsequent versioned
experiments have completed. Its primary
past-normalized ADE, equal-physical-scene aggregation, fit/development roles,
three seeds and policy settings are frozen. Independent calibration and
confirmation remain unresolved; none of these development results clears that
requirement. See [the continuous-context decision](continuous_context_v5_decision.md)
and the [current evidence in the working manuscript](paper_working_draft.md).
The later fit-only repairs preserve the 11,966-window cohort and keep Students,
development, calibration and confirmation roles closed. SDD auxiliary source-role
and sampling remain a separate unresolved choice; the completed diagnostic
image/geometry bridge is not training admission. This draft does not supersede
those executed registrations or describe a currently running training process.

The [executable contract](experiment_contract/implementation_and_limits.md) now checks declared data roles, recursive artifact exposure, exact horizon availability and frozen calibration/confirmation identity. The local draft contains no split assignments or new risk settings. All nine current canonical recordings retain their historical development-use status. Passing a contract check would not establish untouched data, IID, accuracy or a calibrated risk guarantee; formal training must integrate the interface and preserve the underlying source/decision evidence.

## Question

Can a scene-level intervention policy exploit neural motion forecasts while controlling excess prediction risk relative to a strong, fixed baseline, including degradation on easy agents and conflict caused by mixing predictions?

## Proposed Method

For each observed scene X, a fixed causal baseline returns B(X) and a learned forecaster returns N(X). A gate chooses a vector a in [0,1]^agents, giving Y_hat = B + a * (N - B). Discrete {0,1} selection is the initial experiment; blending is a separate ablation.

1. Learn per-agent expected gain and positive harm from out-of-fold train predictions of N and B, not from oracle baseline classes. Normalize with a past-only scale, and retain raw dataset-local errors for evaluation.
2. Construct edges using current/past geometry. Jointly select interventions using predicted excess error plus penalties for candidate-trajectory proximity and group-motion inconsistency. Do not use future labels to construct this graph or runtime constraints.
3. Select architecture and policy candidates on development validation. Freeze the candidate set before independent calibration. Unsupported sources/horizons use the baseline.
4. Evaluate the locked predictor and locked gate once on a confirmatory holdout. Historical test results cannot regain holdout status by renaming them.

## Statistical Target

Primary task metrics must be frozen after the dataset protocol audit: per-domain ADE/FDE over identical valid future samples; project t50 raw-frame results separately. The proposed primary method test is improvement at a fixed intervention-risk budget against the strongest matched baseline and simple independent gating. Easy loss tolerance remains 2% relative to the frozen baseline within each declared evaluation domain. Easy labels may use evaluation outcomes only for reporting; runtime gates must predict difficulty from the past.

Calibration can use a finite policy family and scene-level bounded losses, with an LTT-style simultaneous test. For M fixed policies and K fixed risks, if independent calibration scenes generate Z in [a,b], Hoeffding plus a union bound gives the standard upper bound:

```text
U(lambda, k) = mean_scene Z(lambda, k)
             + (b-a) * sqrt(log(M*K/delta)/(2*n_k))
```

Accept only policies whose relevant bounds meet prespecified tolerances; use the baseline if none pass. This is an application of established tools, not a new theorem. Guarantees require fixed candidates, independent calibration, bounded losses and the stated sampling assumptions. A bound for clipped/normalized loss does not prove a bound for unbounded raw ADE. Groups with few independent scenes may yield vacuous bounds. No arbitrary-shift or collision-free guarantee is claimed.

Use paired scene/source bootstrap with >=2,000 resamples and >=3 training seeds. Report independent cluster counts. Overlapping rows are not independent calibration or bootstrap units.

## Minimal Factorial Experiments

| Experiment | Change | Keep fixed | Decisive observation |
| --- | --- | --- | --- |
| E0 | Rebuild independent protocol | Original recording identity, causal masks | No source/teacher/goal leakage |
| E1 | Oracle-class vs raw-FDE vs realized excess-risk labels | Predictor, features, rows, compute | Excess-risk objective improves risk-coverage tradeoff |
| E2 | No gate vs independent vs scene-uniform vs joint selection | Same candidate forecasts | Joint selection reduces combination conflicts at matched coverage |
| E3 | Row-level vs scene-level calibration | Frozen policy family | Empirical coverage and uncertainty reflect scene dependence |
| E4 | Strong classical selector vs neural risk head | Same supervision and features | Whether neural features are necessary |
| E5 | Context absent / correct / mismatched | Predictor size and split | Whether visual/interaction context contributes predictive value |
| E6 | Frozen backbone with and without gate | Each external forecasting baseline | Wrapper benefit is not specific to one weak forecaster |

## Dataset Prerequisites

- Canonicalize original recordings across OpenTraj, ETH/UCY and TrajNet. Collection names do not establish external independence.
- Check whether cached teacher predictions were trained on rows later moved into new validation/test splits. Refit teachers within split when required.
- Verify baseline future paths. A linearly interpolated endpoint is an explicit comparator, not automatically the full Stage37 trajectory policy.
- Verify FPS/annotation step before seconds claims, and geometry before meter claims. Standard benchmark protocol and raw-frame project protocol remain separate.
- Audit legal/source conditions against actual downloaded artifacts; old blocker reports alone do not establish current availability.

## Submission Decision

By 2026-10-25 require independent positive evidence for the main mechanism, a matched published baseline, meaningful ablations, cluster-aware uncertainty, and an honest limitation section. Strong internal stage gates alone do not satisfy this decision.

No Stage5C execution, SMC, production promotion, paid compute launch, paper submission, or external messaging is authorized by this protocol document.
