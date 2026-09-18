# M3W: Real-World Multimodal Agent-Scene World Model

M3W is my research project on top-down multi-agent world modeling.

The question behind the project is simple:

> If I can see a scene, the agents in it, their recent motion, and their local interactions, can I predict what happens next more reliably than strong causal motion baselines?

I started this repo to answer that question carefully, not just to collect a nice-looking demo. The work here includes the models that improved results, the ones that failed, the leakage checks, the safety rules, and the notes that keep me honest about what the evidence does and does not prove.

## Research Question

My primary task is **eight observed annotation steps to twelve predicted steps**.
Raw-frame `t+50` is a separate supplement. I study when a neural forecast adds
value over a strong causal baseline, how to estimate the harm from switching,
and whether decisions for interacting agents should be made together.

The proposed contribution is baseline-relative, scene-level intervention with
support-aware fallback. A Transformer, JEPA encoder or cost head is not novel
just because it is part of this system. Each component has to earn its place
through matched comparisons and useful out-of-scene results.

## Current Evidence

The implementation runs, but the clean development experiments have **not yet
established a deployable neural advantage or a submission-ready method**.

| Question | What the completed evidence shows |
| --- | --- |
| Do neural trajectory models beat strong motion baselines? | The fixed three-seed Transformer and K=1 EqMotion comparisons did not produce safe positive gains on the primary task. |
| Does longer training help? | Learning-rate decay produces a small source-training gain, but it does not transfer to the excluded source scene. |
| Does the tested RGB representation help? | The matched source comparison is negative. More input modalities are not automatically more predictive information. |
| Does cost-aware fallback help? | It reduces neural harm, but the fixed source readout still loses 1.246% to stationary CV. The unprotected control loses 1.744%. |
| Do scene-excluded candidate forecasts remain useful? | Twelve fresh fits all lose on their excluded site; equal-site gain is -5.016%. Fixed candidate/CV oracle headroom is below 0.53%, so another gate alone is not the next repair. |
| Does removing the static-target loss repair them? | No. Twelve matched new fits increase oracle headroom to 3.760%, but actual gain is -98.719% and static-target harm is much larger. |
| Do raw annotation checks and past-box features explain the failure? | Small changes are common, but >10px queries contribute 53.25% of baseline error and still lose. Forty-eight fixed probability probes find no stable added-box benefit. |
| Do pretrained image features repair source transfer? | No. Thirty-six matched trajectory heads complete 360,000 updates. Geometry/current-image/eight-frame gains are -0.070%/-1.908%/-6.102%; all held fits are negative. |
| Are the historical external selector gains independently verified? | No. Recording duplication, teacher exposure and test-based selection make those scores exploratory. |
| Is scene-level joint intervention validated? | The implementation and matched-count controls exist; a reliable advantage and independent risk calibration remain unproved. |

The latest [fixed deferral readout](outputs/publication_readiness_2026_09/source_deferral_transfer_v1/conclusions.md)
retains all six trained endpoints and three matched controls. All three
cost-supervised seeds lose to CV; the conditional recording interval is
[-4.430%, -0.567%]. It concerns seven recordings of **one previously explored
site**, not independent confirmation. Exact replay verifies reproducibility,
not forecasting quality. Complete rejection returns the baseline and is not
a new prediction success.

The latest [candidate cross-fit experiment](outputs/publication_readiness_2026_09/source_crossfit_v1/conclusions.md)
completed all 120,000 updates across four internal site folds and three seeds.
Equal-site gain is -5.016%, conditional interval [-8.397%, -2.488%]. Most excess
error comes from predicted movement on stationary targets, but the remaining
moving-target predictions also lose on average. The experiment isolates producer
exposure, not the causal reason for the transfer gap. Bookstore and the main
evaluation remain unscored; no new model is deployed.

The [matched loss intervention](outputs/publication_readiness_2026_09/source_motion_candidate_v1/conclusions.md)
has now completed another 120,000 updates. Removing static-target gradients
makes the forecast less conservative, but it also worsens moving-target error.
Rotating its predictions retains most of the oracle headroom, so that headroom
alone is not evidence of accurate motion direction or usable neural dynamics.
All twelve models replay exactly; the scientific result is still negative.

The completed [motion-quality diagnostic](outputs/publication_readiness_2026_09/source_motion_quality_v1/conclusions.md)
aligns all 15,430 source queries to raw annotations. It distinguishes tiny
coordinate changes from larger excursions without deleting either group.
Past-box features do not repair cross-site motion probabilities. Interpolation
controls after the query also occur in 15,316 histories, reinforcing the
offline-annotation limitation rather than establishing real-time perception.

The completed [pretrained temporal comparison](outputs/publication_readiness_2026_09/source_pretrained_temporal_v1/conclusions.md)
adds frozen visual features without changing the cohort, loss or sampling budget.
Appearance improves training fit slightly but worsens excluded-scene prediction.
Eight-frame appearance loses another 4.194 percentage points relative to current
appearance. All 36 heads replay exactly; this confirms the negative result, not
a deployable visual dynamics contribution.

## Evidence and Reproduction

The detailed record is kept separately so that the project overview remains
readable:

- [Results ledger](README_RESULTS.md): complete experiment outcomes, failures and current evidence boundaries.
- [September research history](README_RESEARCH_HISTORY_2026_09.md): the detailed routes and diagnoses behind this summary.
- [Recording and teacher-lineage audit](outputs/publication_readiness_2026_09/recording_lineage_audit.md): why historical external gains cannot be treated as independent evidence.
- [Working paper](outputs/publication_readiness_2026_09/paper_working_draft.md): the research question, method proposal, results and missing evidence, not a finished submission.
- [Latest experiment reproduction](outputs/publication_readiness_2026_09/source_pretrained_temporal_v1/reproducibility.md): commands, hashes, replay checks and limitations.
- [Data-role contract](outputs/publication_readiness_2026_09/experiment_contract/implementation_and_limits.md): training, selection, calibration and confirmation boundaries.

The current observation contract uses supplied historical annotations. Some
annotations may have been interpolated using later controls; past-indexed
access therefore does not prove strict sensor-as-of availability. Future
targets are kept out of inference features, and previously explored scenes
cannot become independent tests by renaming their roles.

## What The System Looks At

The current M3W pipeline works with dataset-local top-down trajectories. It uses information that would be available at inference time:

- recent agent history;
- speed, acceleration, heading, curvature, and stop/go behavior;
- neighbor density and interaction signals;
- train-only scene or goal context when that context is legally available;
- causal baseline rollouts;
- dataset, scene, horizon, and domain metadata;
- risk heads for failure, gain, harm, and fallback decisions.

I also maintain a neural track with Transformer dynamics, JEPA-style representation learning, hybrid heads, waypoint prediction, and protected residual policies. Guarded selection, causal history windows, full-waypoint structure, domain-aware routing, and safety floors are the most promising routes in the historical experiments. Their external gains remain exploratory until the clean evaluation is complete; neither the selector nor the neural branch has earned a new deployment claim from this audit.

## What This Repo Is For

This repository is a research record. The most important rule in the project is that a result has to survive the boring checks: no future leakage, no test endpoint goals, no central-velocity shortcuts, no easy-case damage hidden inside aggregate gains, and no metric claims without calibration.

For a quick orientation:

| File or directory | What to read it for |
| --- | --- |
| [`README_RESULTS.md`](README_RESULTS.md) | Detailed results ledger and current evidence boundaries. |
| [`README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md`](README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md) | Chinese long-form summary of routes tried, failures, causes, and successes. |
| [`research_state.json`](research_state.json) | Machine-readable snapshot of the current project state. |
| `outputs/m3w_neural_v1/` | Neural world-model reports and model-card style summaries. |
| `outputs/stage42_long_research/` | Cross-domain safety, replay, full-waypoint, and paper-claim evidence. |
| `outputs/stage43_latent_state/` | Latent-state, graph/history/context, and reviewer-style validation reports. |

Large datasets, caches, checkpoints, videos, images, third-party data, and local virtual environments are intentionally kept out of git.

## What I Am Not Claiming

M3W is not a true 3D world model yet. It is not a foundation world model. SDD results are pixel-space unless calibration is verified. External results are dataset-local unless their geometry is verified. `t+50` and `t+100` are raw annotation-frame horizons, not seconds. Self-audited or inferred labels are not human gold labels.

Stage5C latent generative execution has not been enabled. SMC has not been enabled.

The current claim is narrower: this repo contains a protected 2.5D multi-agent world-state research system and an active neural dynamics track. Its historical external evaluation has identified independence failures that I am repairing before making new generalization or deployment claims. This external audit does not establish the status of every SDD experiment.

## Running Locally

On Apple Silicon, training should use the arm64 PyTorch environment:

```bash
.venv-pytorch/bin/python
```

Focused checks for the new data and evaluation path:

```bash
.venv-pytorch/bin/python -m pytest tests/test_m3w_deferral_development.py -q
.venv-pytorch/bin/python -m pytest tests/test_m3w_cost_sensitive_deferral.py tests/test_m3w_matched_coverage.py tests/test_m3w_citr_recordings.py tests/test_m3w_confirmation_evaluation.py tests/test_m3w_risk_calibration.py tests/test_m3w_eqmotion_adapter.py tests/test_m3w_development_evaluation.py tests/test_m3w_supervised_intervention.py tests/test_m3w_external_source_audit.py tests/test_m3w_experiment_contract.py tests/test_m3w_joint_intervention.py tests/test_m3w_causal_recordings.py tests/test_m3w_recording_lineage.py tests/test_stage44_worldcore.py
```

The legacy full suite (`python -m pytest tests`) includes integration training and can rewrite reports in the working directory. It is not yet an isolated, read-only smoke test; preserve existing experiment outputs before running it. The [local/CREATE runbook](outputs/publication_readiness_2026_09/local_create_runbook_zh.md) records the verified environment and recovery checks.

Training scripts are written around checkpointing, heartbeat logs, resume support, CPU/MPS-safe execution, and single-process dataloading.

## Next Step

Improve candidate utility before fitting another risk head. The loss control
has separated larger oracle headroom from actual predictive value; it has not
supplied a safe candidate. The annotation-scale check is now complete: neither
small changes alone nor the tested past-box features explain or repair the
transfer failure. Next I will test a different past-information representation,
after checking reusable visual assets and defining a matched geometry control.
This is not a reason to repeat failed compact RGB routes unchanged.
OOF labels do not automatically permit a second-level
validation split: every upstream producer must also exclude the risk head's
validation scene.
[Provenance boundary](outputs/publication_readiness_2026_09/source_crossfit_v1/method_and_limits.md).

The [fixed loss-control registration](outputs/publication_readiness_2026_09/source_motion_candidate_decision.md)
and all negative outcomes remain available. No new policy has been deployed,
and no test threshold was changed to rescue this result.

The [registered information audit](outputs/publication_readiness_2026_09/source_motion_quality_decision.md)
is complete: 48 fresh fits, exact replay and 44 focused checks. Its negative
probability results do not change the main task or justify a new deployment.

The [registered pretrained comparison](outputs/publication_readiness_2026_09/source_pretrained_temporal_decision.md)
is complete and negative. The next diagnosis concerns transferable information
at the source-image boundary, not another threshold sweep. Existing native-detail
and optical-flow negative controls remain relevant; simply adding resolution or
more modules is not an established repair.

The larger goal is unchanged: demonstrate useful neural dynamics, compare
independent and joint intervention at matched coverage, preserve easy cases,
and obtain genuinely independent calibration and confirmation. More overlapping
windows cannot substitute for more independent scenes. I am working toward
CVPR 2027, not claiming that implementation progress guarantees a publishable
result or acceptance.

When a route fails, I keep the evidence. A successful method must show where
it improves the baseline, where it does not, and how the result can be reproduced.
