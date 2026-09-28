# M3W: Real-World Multimodal Agent-Scene World Model

M3W is my research project on top-down multi-agent world modeling. I study
when a neural forecast adds value over a strong motion baseline, and how to
avoid harming the cases that the baseline already handles well.

The long-term aim is a useful agent-scene world model. The current work is
narrower: causal forecasting, learning the gain and harm of an intervention,
and deciding which agents should receive a neural prediction.

## Research Question

The main observation protocol is **8 observed steps and 12 predicted steps**.
The current European-source experiment uses a stride of 12 raw frames;
raw-frame `t+50` is a separate historical supplement, not the same task.

I am testing three linked questions:

1. Can the neural predictor contribute something beyond a strong causal baseline?
2. Can past-only context identify where switching helps and where it hurts?
3. Can joint decisions improve on independent selection at the same intervention count?

JEPA, Transformer and hybrid models are part of the research track. Combining
these modules is not, by itself, evidence of a useful world model or a new method.

## Current Result

**Better easy-case probabilities did not produce better neural selection.**

I completed a paired experiment with 216 risk heads, keeping the forecasts,
causal inputs, initialization and training queries fixed. One arm learned signed
risk directly; the other also learned easy-case occurrence and conditional costs.
All 108 action groups were frozen in `96b82ac3` before development readout.

| Same-count supervised versus marginal control | Result | Nominal 95% locality interval |
|---|---:|---:|
| ADE improvement | -0.02047% | [-0.03206%, -0.00957%] |
| Positive harm reduction, full-floor denominator | +0.00649 pp | [+0.00216, +0.01314] |

Easy-occurrence Brier improved from 0.2621 to 0.1581. However, avoided harm was
smaller than the benefit lost by the new allocation. The common-count control
rules out switching less as the only explanation. Easy error stayed within the
existing limit, but 51 of 216 dependent views violated selected risk and 24 had
undefined selected risk because they entirely abstained. **No deployment upgrade.**

Training completed on CREATE: 216 heads, 2,000 updates each. Checkpoint hashes,
paired sampling and an exact first-pair training replay passed. All 108 action
groups and the full readout replay exactly. Independent arithmetic checks and
51 scoped tests pass; they verify this negative result, not model efficacy. These
results use twelve already-opened development localities, not independent
confirmation. I am retaining the failed comparison rather than changing its
threshold or replacing the original risk denominator.

The fitting-only diagnostic now reproduces exactly. At the final supervised
states, auxiliary shared-gradient norms have a median ratio of 213 to direct
risk; 60 of 432 repeated batches have an opposing total gradient. This supports
testing a fixed auxiliary norm cap, not declaring the cause or repair proven.
No new held outcomes or independent sources were opened. The next paired fit
is registered to change only this gradient cap, with all108groups and2000updates
per head retained; the original negative result stays frozen.

- [Current paired results and all controls](outputs/publication_readiness_2026_09/european_easy_hurdle_v1/results.md)
- [Benefit/harm accounting and actual training losses](outputs/publication_readiness_2026_09/european_easy_hurdle_v1/failure_analysis.md)
- [Locality, seed and tail results](outputs/publication_readiness_2026_09/european_easy_hurdle_v1/locality_seed_quality.md)
- [Training and exact replay evidence](outputs/publication_readiness_2026_09/european_easy_hurdle_v1/training_result.md)
- [Final verification scope](outputs/publication_readiness_2026_09/european_easy_hurdle_v1/verification_report.md)
- [Decision, limitations and next diagnostic](outputs/publication_readiness_2026_09/european_easy_hurdle_v1/conclusions.md)
- [Actual gradient measurements and limits](outputs/publication_readiness_2026_09/european_easy_gradient_diagnostic_v1/results.md)
- [Controlled repair rationale](outputs/publication_readiness_2026_09/european_easy_gradient_diagnostic_v1/conclusions.md)
- [Registered paired repair](outputs/publication_readiness_2026_09/european_easy_risk_priority_v1/protocol.md)

## Previous Controlled Result

**Correcting average risk bias did not produce a better deployment policy.**

After training 216 risk heads, I tested whether their remaining fitting-only
score bias explained unsafe switching. The forecasts and learned utilities
were frozen; only two already-fitted risk offsets changed. All 108 action groups
were committed before the new development readout.

The offsets greatly reduced intervention, but also removed useful predictions.
At exactly the same intervention count in each query, centered risk was worse
than the original-risk control:

| Frozen risk head | ADE gain versus same-count control | Nominal 95% locality interval |
|---|---:|---:|
| Pointwise | -0.00335% | [-0.00882%, -0.00017%] |
| Subset aggregate | -0.00523% | [-0.01182%, -0.00043%] |

For the aggregate head, intervention fell from 7.80% to 0.67%. Two hundred of 216
dependent views then entirely abstained; six remaining views still violated
the 2% selected-harm screen. Easy cases were preserved, but this is **not a safe
deployment upgrade**. Lower fitting loss and fewer harmful switches are not
enough when useful switches disappear too.

All 108 action groups and their evaluation replay exactly. **34 scoped tests
pass**; separate arithmetic checks cover 1,495,800 query/head constraints,
3,456 cost views and 23,714 locality reductions. These repeated contexts are
not independent samples. A replay-only JSON identity-format defect was fixed
without changing original code or scientific outputs.

Most rejected original admissions triggered the easy-risk constraint. This
motivated the paired easy-occurrence experiment above. It improved probability
and conditional-cost fit, but did not repair allocation or selected risk.
Independent selection, calibration and confirmation remain closed.

- [Centered-risk experiment and failure analysis](outputs/publication_readiness_2026_09/european_centered_risk_policy_v1/conclusions.md)
- [New paired repair: audit, execution status and remaining work](outputs/publication_readiness_2026_09/european_easy_hurdle_v1/results.md)
- [Completed training and exact replay evidence](outputs/publication_readiness_2026_09/european_easy_hurdle_v1/training_result.md)
- [All controls, intervals and risk failures](outputs/publication_readiness_2026_09/european_centered_risk_policy_v1/results.md)
- [Frozen verification record](outputs/publication_readiness_2026_09/european_centered_risk_policy_v1/verification.json)
- [Current reproduction guide](outputs/publication_readiness_2026_09/european_centered_risk_policy_v1/operation_zh.md)

- [What the frozen decisions reveal](outputs/publication_readiness_2026_09/european_selected_risk_diagnosis_v1/conclusions.md)
- [Residual and support results](outputs/publication_readiness_2026_09/european_selected_risk_diagnosis_v1/results.md)
- [Diagnostic verification](outputs/publication_readiness_2026_09/european_selected_risk_diagnosis_v1/verification.json)
- [Fitting-only bias probe](outputs/publication_readiness_2026_09/european_signed_bias_probe_v1/conclusions.md)
- [Bias-fit results and verification](outputs/publication_readiness_2026_09/european_signed_bias_probe_v1/results.md)
- [Parent neural experiment](outputs/publication_readiness_2026_09/european_subset_excess_v1/results.md)
- [Chinese reproduction guide](outputs/publication_readiness_2026_09/european_selected_risk_diagnosis_v1/operation_zh.md)

The earlier fixed-predictor allocation experiment did improve ADE at matched
counts, but failed observed-risk control. The subsequent pure query-aggregate
loss also failed. I retain these negative results because they distinguish
better allocation, better prediction loss and reliable risk control.

## Evidence Boundaries

The project currently remains a **2.5D multi-agent trajectory/world-state
research system**, not a true 3D or foundation world model. Coordinates are
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
under Rosetta. Current risk-head experiments use 4 compute threads, 1 interop
thread and 0 DataLoader workers, with atomic checkpoints, heartbeat and resume.

A focused check for the latest experiment:

```sh
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_centered_risk_policy.py tests/test_m3w_centered_risk_verification.py tests/test_m3w_centered_identity_replay.py
```

The reproduction guide above documents the frozen fit, action and evaluation
sequence. Existing sealed outputs are checked, not silently overwritten.
The legacy full test suite includes integration training and can rewrite
reports; it is not an isolated read-only smoke test.

## Next Question

Which conditional errors make useful and harmful interventions hard to separate
across sources? The global-offset experiment now has a negative answer even at
matched counts. I will first inspect easy-risk exposure and error scale within
source-excluded fitting data. Any repair must preserve benefit and easy
protection, retaining an original-score control at the same
intervention count, freezing decisions before readout. No conditional repair
has been trained yet. I will not tune thresholds on these diagnostics or open independent
confirmation data to rescue the method.

The larger goal remains useful neural dynamics with reproducible, independent
evidence. More stages, more overlapping windows or a lower training loss are
not substitutes for that result.
