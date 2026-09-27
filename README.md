# M3W: Real-World Multimodal Agent-Scene World Model

M3W is my research project on top-down multi-agent world modeling. I study
when a neural forecast adds value over a strong motion baseline, and how to
avoid harming the cases that the baseline already handles well.

The long-term aim is a useful agent-scene world model. The current work is
narrower: causal forecasting, learning the gain and harm of an intervention,
and deciding which agents should receive a neural prediction.

## Research Question

The main observation protocol is **8 observed steps and 12 predicted steps**.
The current European-source experiment uses a stride of12 raw frames;
raw-frame `t+50` is a separate historical supplement, not the same task.

I am testing three linked questions:

1. Can the neural predictor contribute something beyond a strong causal baseline?
2. Can past-only context identify where switching helps and where it hurts?
3. Can joint decisions improve on independent selection at the same intervention count?

JEPA, Transformer and hybrid models are part of the research track. Combining
these modules is not, by itself, evidence of a useful world model or a new method.

## Current Result

**Anchored subset-risk training did not repair safe selection.**

I trained216 matched risk heads with432,000updates. Both arms use identical
features, initialization, query and row draws, and optimizer budgets. They
differ only in how an auxiliary loss supervises three causal selection subsets.
The trajectory predictors, protected floor and utility model stay frozen.

At the same per-query intervention count, aggregate subset supervision gains
**0.00068% ADE [-0.00870%, +0.01049%]** over its pointwise control. That does
not support an accuracy advantage. Its total-reference harm diagnostic
**increases by0.00228percentage points [0.00014,0.00472]**.

The joint policy gains0.5079%ADE over the protected floor, but84of216 dependent
views exceed the selected-harm screen and16ratios are undefined. Easy net error
is preserved; this is still **not a deployment upgrade**. A tiny secondary
gain against an older ranking control does not change that conclusion.

These are results from twelve already-opened development localities, with
three forecasting seeds and3,000locality-bootstrap draws. Training and actions
were committed before readout. Full replay and independent verification are
in progress. Independent selection, calibration and confirmation remain closed.

- [Experiment conclusions](outputs/publication_readiness_2026_09/european_subset_excess_v1/conclusions.md)
- [Failure analysis](outputs/publication_readiness_2026_09/european_subset_excess_v1/failure_analysis.md)
- [Registered protocol](outputs/publication_readiness_2026_09/european_subset_excess_v1/protocol.md)
- [Method positioning and limits](outputs/publication_readiness_2026_09/european_subset_excess_v1/method_positioning.md)
- [Chinese reproduction guide](outputs/publication_readiness_2026_09/european_subset_excess_v1/operation_zh.md)

The earlier fixed-predictor allocation experiment did improve ADE at matched
counts, but failed observed-risk control. The subsequent pure query-aggregate
loss also failed. I retain these negative results because they distinguish
better allocation, better prediction loss and reliable risk control.

## Evidence Boundaries

The project currently remains a **2.5D multi-agent trajectory/world-state
research system**, not a true3D or foundation world model. Coordinates are
image-local, pixel-space or dataset-local unless separately verified. Raw-frame
horizons are not seconds. Detector, inferred and self-audited labels are not
human gold, and statistical harm diagnostics are not physical-safety guarantees.

Future targets are used for loss and evaluation, never as inference features.
Central velocity and test-endpoint goals are excluded. Source exclusion applies
to upstream predictors, preprocessing, teachers and downstream heads. Supplied
historical annotations can still have interpolation limitations: past-indexed
access alone does not establish strict sensor-as-of availability.

Earlier external Stage35/37/43/44 scores have recording, teacher-exposure or
selection limitations. They remain exploratory; renaming their splits cannot
make them independent evidence. Current percentages are not directly
comparable with those historical `t+50` results.

Stage5C latent-generative execution and SMC remain disabled. No new deployment,
calibration certificate or submission-readiness claim follows from this round.

## Reading the Repository

| Record | Purpose |
|---|---|
| [Results ledger](README_RESULTS.md) | Detailed outcomes, including negative results and verification status |
| [Research state](research_state.json) | Current machine-readable status and next action |
| [Archived research history](README_RESEARCH_HISTORY_2026_09_27.md) | Complete earlier project overview, preserved before this shorter introduction |
| [Earlier September history](README_RESEARCH_HISTORY_2026_09.md) | Earlier routes and diagnoses |
| [Recording/teacher audit](outputs/publication_readiness_2026_09/recording_lineage_audit.md) | Limits of historical external claims |
| [Data-role contract](outputs/publication_readiness_2026_09/experiment_contract/implementation_and_limits.md) | Training, selection, calibration and confirmation boundaries |
| [Working paper](outputs/publication_readiness_2026_09/paper_working_draft.md) | Research draft, not a finished or submission-ready paper |
| [Local/CREATE runbook](outputs/publication_readiness_2026_09/local_create_runbook_zh.md) | Environment, scheduling and recovery guidance |

Raw datasets, feature/history/latent caches, large checkpoints, videos,
third-party images and local virtual environments stay out of Git.

## Running Locally

On Apple Silicon, use the native arm64 PyTorch environment, not Intel Conda
under Rosetta. Current risk-head experiments use4compute threads,1interop
thread and0DataLoader workers, with atomic checkpoints, heartbeat and resume.

A focused check for the latest experiment:

```sh
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_subset_excess.py tests/test_m3w_subset_excess_verification.py
```

The reproduction guide above documents the frozen fit, action and evaluation
sequence. Existing sealed outputs are checked, not silently overwritten.
The legacy full test suite includes integration training and can rewrite
reports; it is not an isolated read-only smoke test.

## Next Question

Why does better risk prediction on fixed proxy groups fail to control harm on
the groups actually chosen by the optimizer? The next step is a frozen-action
residual and support diagnosis, followed by one preregistered targeted repair.
I will not tune thresholds on these readouts or open independent confirmation
data to rescue the method.

The larger goal remains useful neural dynamics with reproducible, independent
evidence. More stages, more overlapping windows or a lower training loss are
not substitutes for that result.
