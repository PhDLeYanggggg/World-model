# M3W: Real-World Multimodal Agent-Scene World Model

M3W is my research project on top-down multi-agent world modeling.

The question behind the project is simple:

> If I can see a scene, the agents in it, their recent motion, and their local interactions, can I predict what happens next more reliably than strong causal motion baselines?

I started this repo to answer that question carefully, not just to collect a nice-looking demo. The work here includes the models that improved results, the ones that failed, the leakage checks, the safety rules, and the notes that keep me honest about what the evidence does and does not prove.

## Current Evidence Status

The protected selector remains the historical reference implementation. It starts from causal motion baselines and switches only when the expected gain is large enough and the estimated easy-case risk is low enough. I am not currently treating its external results as independently validated deployment evidence.

During the September 2026 publication audit, I found byte-identical recordings under different dataset paths. The Stage35/37 validation and test sets share 47,223 cached windows, and the later Stage43/44 split has train/test duplication as well as inherited teacher-training exposure. Stage37 and the original Stage44 code also used test metrics to rank model variants. These issues mean the external claims need a clean rerun. The [recording audit](outputs/publication_readiness_2026_09/recording_lineage_audit.md) and [original split audit](outputs/publication_readiness_2026_09/stage35_recording_lineage_audit.md) document the evidence.

For traceability, these are the **historically reported Stage37 numbers, not corrected confirmatory results**:

| Slice | Result |
| --- | ---: |
| Overall improvement | +13.48% |
| Raw-frame `t+50` improvement | +8.46% |
| Hard/failure improvement | +15.54% |
| Easy-case degradation | 0.041% |
| `t+50` bootstrap CI | [+7.69%, +9.15%] |

The table is retained so that older reports remain interpretable. Its bootstrap interval does not account for duplicate recordings or test-based model selection. I have repaired the WorldCore selection path to use validation only and added a training preflight that rejects the unchanged legacy caches. This is a protocol repair, not a new model improvement; the historical weights have not been replaced.

I have also rebuilt a recording-centric reader directly from the original positions, without inherited teacher outputs. It keeps past inputs separate from future labels, groups duplicate dataset packaging, and requires exact target timestamps. The [rebuild and causal-access checks](outputs/publication_readiness_2026_09/causal_recording_checks.md) are engineering evidence, not a new benchmark result. The formal evaluation protocol still needs to be frozen before retraining.

The next method asks a more specific question: when several agents may switch away from a baseline, should those decisions be made together? I have implemented a [joint intervention prototype](outputs/publication_readiness_2026_09/joint_intervention/method_and_checks.md) with explicit predicted-risk and intervention budgets. Its optimizer and coordinate handling pass targeted checks, but it has not yet demonstrated better prediction or calibrated real-world risk. Joint trajectory modeling already has substantial prior work; the contribution will need to come from the matched experiments, not the presence of a joint optimizer.

I have added an [exact-coverage control](outputs/publication_readiness_2026_09/matched_coverage/method_and_limits.md) to test that distinction: independent and joint policies must switch the same number of agents in each scene query. It prevents a policy that simply switches less from being credited with better coordination. The control is diagnostic, not deployable, and its real accuracy comparison is pending. [Regression-deferral prior work](outputs/publication_readiness_2026_09/joint_intervention/deferral_and_coverage_prior_work.md) also makes clear that learning when to use another predictor is not a novelty claim by itself.

For the clean rerun, I now keep data roles and learned-artifact provenance in a [hash-bound experiment contract](outputs/publication_readiness_2026_09/experiment_contract/implementation_and_limits.md). It checks upstream teacher exposure and freezes the evaluated model family before calibration or confirmation. The current protocol is deliberately unapproved: previously explored recordings remain development material, and the remaining scientific choices are not filled in just to make training start.

I am also checking what the other local datasets can genuinely add. The [external source review](outputs/publication_readiness_2026_09/external_source_audit/source_review.md) distinguishes synchronized scenes from isolated tracks, repeated controlled trials and duplicate representations. More trajectory points do not automatically mean more independent evidence; none of these sources has been promoted to a new confirmation set or used to claim a fresh gain.

The local [CITR diagnostic reader](outputs/publication_readiness_2026_09/citr_causal_intake/implementation_and_limits.md) now preserves 95,648 original positions across 38 controlled clips, with separate pedestrian and vehicle identities and an exact map back to each CSV row. This gives me synchronized mixed-agent examples for future mechanism checks. All clips still belong to one physical site; they are not 38 new independent test scenes, and I have not used this conversion to claim a forecasting gain.

The rebuilt reader now connects to a [resumable forecasting and cost-learning backend](outputs/publication_readiness_2026_09/supervised_backend/implementation_and_limits.md). It learns baseline-relative benefit and harm from predictions made outside each producer's training fold. A [development evaluator](outputs/publication_readiness_2026_09/development_evaluation/implementation_and_limits.md) compares five intervention controls using the same predictions, retains agents with missing future labels in the decisions, and reports uncertainty by physical scene. I have checked this connection on synthetic examples and the input boundary on real trajectories. Those checks establish an implementation, not predictive improvement: clean real-data fitting, development selection and independent confirmation are still pending.

To avoid comparing only against my own networks, I have also connected a [version-pinned EqMotion core](outputs/publication_readiness_2026_09/public_baselines/compatibility_and_limits.md) to the causal reader and training path. The current adapter emits one fixed head and is explicitly a K=1 adaptation, not a reproduction of the paper's best-of-20 result. Source integrity, CPU/MPS training recovery and past-only input behavior are checked; its real forecasting comparison is still pending. The published model's preprocessing, sampling budget and checkpoint selection need the same scrutiny as my own code.

I now have a [frozen-policy risk-screening path](outputs/publication_readiness_2026_09/risk_calibration/implementation_and_limits.md), but implementation is not a safety guarantee. The current rebuilt collection contains only six physical-scene groups, all previously used during development. Even an optimistic calculation shows that the present conservative bound would be too wide for a useful small-risk claim. I will keep empirical prediction gains, statistical assumptions and physical safety as separate questions; duplicating windows cannot close that evidence gap.

The [final comparison path](outputs/publication_readiness_2026_09/confirmation_evaluation/implementation_and_limits.md) keeps every training seed visible and freezes the compared policies before final labels are read. It reports scene-level uncertainty, per-seed easy-case damage and matched joint-control comparisons without choosing a winner on the final set. The complete connection has been exercised on synthetic data; it does not replace the still-pending real independent experiment.

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
.venv-pytorch/bin/python -m pytest tests/test_m3w_matched_coverage.py tests/test_m3w_citr_recordings.py tests/test_m3w_confirmation_evaluation.py tests/test_m3w_risk_calibration.py tests/test_m3w_eqmotion_adapter.py tests/test_m3w_development_evaluation.py tests/test_m3w_supervised_intervention.py tests/test_m3w_external_source_audit.py tests/test_m3w_experiment_contract.py tests/test_m3w_joint_intervention.py tests/test_m3w_causal_recordings.py tests/test_m3w_recording_lineage.py tests/test_stage44_worldcore.py
```

The legacy full suite (`python -m pytest tests`) includes integration training and can rewrite reports in the working directory. It is not yet an isolated, read-only smoke test; preserve existing experiment outputs before running it. The [local/CREATE runbook](outputs/publication_readiness_2026_09/local_create_runbook_zh.md) records the verified environment and recovery checks.

Training scripts are written around checkpointing, heartbeat logs, resume support, CPU/MPS-safe execution, and single-process dataloading.

## Next Step

I am developing this work toward a CVPR 2027 submission on when neural motion predictions can safely improve a strong baseline. The next study focuses on baseline-relative risk and joint intervention across agents. My [research direction and evidence audit](README_M3W_INNOVATION_AND_CVPR2027_ZH.md) explains the proposed contributions, the limitations of the current experiments, and the remaining comparisons. In particular, the latest WorldCore architecture ranking used test metrics, so those results remain exploratory until independently confirmed.

The next research step is to make the neural branch contribute something the protected policy does not already provide.

That means:

1. rebuild recording-disjoint evaluation and refit the protected selector within each training fold;
2. promote neural dynamics only if they improve overall, `t+50`, or hard/failure slices without damaging easy cases;
3. keep testing whether scene, goal, graph, and latent context add measurable lift;
4. keep raw-frame and dataset-local claims separate from metric or physical-world claims.

When a route fails, I leave it in the record. When a route works, I want the repo to show exactly where it works, why it is allowed, and what it still does not prove.
