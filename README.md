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

## Latest Diagnostic

I traced where useful neural interventions are lost using all 108 frozen fitting
contexts. Under the registered accounting order, **75.33% of available positive
benefit lies behind the all-risk screen**, and 15.44% behind a nonpositive utility
score. This points beyond the easy-occurrence branch tested most recently.

Removing those screens is not a solution. The nonpositive-utility population
has negative net gain, and useful switches in the all-risk-rejected population
are mixed with substantial positive harm. Utility-only ordering also worsens the
existing sign screen in 54,500 of 168,323 informative fitting query occurrences.

The full diagnostic and exact replay are complete; 38 scoped tests and 19,224
accounting checks pass. These are **in-sample diagnostic findings**, not new
generalization results, a calibrated safety guarantee or a deployment upgrade.
The next controlled test concerns causal gain/harm separation within fitting
sources. Independent selection, calibration and confirmation data remain closed.

- [Diagnostic results and exclusions](outputs/publication_readiness_2026_09/european_fitting_switch_diagnostic_v1/results.md)
- [Interpretation and next test](outputs/publication_readiness_2026_09/european_fitting_switch_diagnostic_v1/failure_analysis.md)

## Last Model Comparison

The next registered experiment tests gain/risk separation between the two fitting
localities using identical inputs and losses, comparing affine logits with a
small nonlinear head. It has not produced a result yet. Independent evaluation
sources and the deployed policy are unchanged.
[Internal-transfer protocol](outputs/publication_readiness_2026_09/european_inner_separability_v1/protocol.md).

**Freezing easy-occurrence probabilities did not improve neural selection.**

I tested whether keeping an existing occurrence estimate fixed would protect it
while learning conditional costs. Both arms used identical split networks, warm
starts and training queries. The experiment trained 216 heads for 2,000 updates
each, then froze every decision before reading development outcomes.

| Same-count comparison | ADE improvement | Nominal 95% locality interval |
|---|---:|---:|
| Fixed versus trainable occurrence | -0.01639% | [-0.02903%, -0.00597%] |
| Fixed versus original raw-risk control | -0.05500% | [-0.10730%, -0.00443%] |

The fixed head had slightly better held-development probability and risk-error
point estimates, but selected worse predictions. It avoided 0.00529 percentage
points of positive harm while losing 0.02146 points of benefit on the common
full-floor diagnostic denominator. That tradeoff does not repair selected risk:
89 of 216 dependent views still violate the risk screen, and 22 remain undefined.
Easy preservation passes the developmental check. **No deployment upgrade.**

All 108 action groups and the complete readout replay exactly; 68 scoped tests
pass. A null optimizer certificate was handled through the existing checked
fallback, preserving the original 95 completed groups and sealed algorithms.
This repairs an engineering interruption, not the optimizer's underlying numerical
failure or the model's risk estimates. Independent confirmation remains closed.

This result narrows the next question: how to retain beneficial interventions
without admitting false-safe ones. Better average probability fit alone is not
enough. These remain developmental risk-controller experiments, not a new world
dynamics result, a safety certificate or a submission-ready system.

- [Result, interpretation and next question](outputs/publication_readiness_2026_09/european_fixed_occurrence_v1/conclusions.md)
- [All controls, metrics and gates](outputs/publication_readiness_2026_09/european_fixed_occurrence_v1/results.md)
- [Benefit/harm accounting](outputs/publication_readiness_2026_09/european_fixed_occurrence_v1/failure_analysis.md)
- [Verification scope](outputs/publication_readiness_2026_09/european_fixed_occurrence_v1/verification_report.md)
- [Chinese run and recovery guide](outputs/publication_readiness_2026_09/european_fixed_occurrence_v1/operation_zh.md)

## Earlier Risk-Priority Result

**A small accuracy recovery did not repair selected risk.**

I tested whether limiting auxiliary gradients could improve risk-constrained
neural selection. The paired experiment trained 216 risk heads for 2,000 updates
each, keeping forecasts, architecture, initialization and sampled queries fixed.
Only the auxiliary-gradient norm cap changed. Every action was frozen before
the development readout.

| Same-count risk-priority comparison | ADE improvement | Nominal 95% locality interval |
|---|---:|---:|
| Versus uncapped auxiliary training | +0.00375% | [+0.00129%, +0.00681%] |
| Versus the original raw-risk control | -0.01435% | [-0.02330%, -0.00611%] |

The repair retained slightly more useful switches, but also increased positive
harm. Selected-risk violations rose from 51 to 54 of 216 dependent views; both
matched policies had 25 views with undefined selected risk. Easy preservation
passed its developmental screen, but the risk screen did not. The original
raw-risk control remains more accurate at matched counts. **No deployment upgrade.**

This followed an earlier negative result: explicit easy-occurrence supervision
improved probability estimates but worsened selection. A fitting-only diagnostic
then measured auxiliary-gradient dominance. The new experiment tests one fixed
response to that imbalance; it does not establish that optimization imbalance
explains the remaining generalization or calibration failures.

All 108 action groups and the complete readout replay exactly. Independent
arithmetic checks and 47 scoped tests pass. All uncapped training controls
reproduce the previous states; the first repaired pair was retrained for exact
replay. These checks support reproducibility, not efficacy. The experiment uses
twelve already-opened development localities, not independent confirmation.
Undefined risk, adverse slices and negative comparisons remain in the results.

I also checked why the repair underperformed on its own fitting data. Full
factor accounting shows a2.11% increase in the risk objective: occurrence
estimation worsened, while changes in the cost factors partly cancelled its
error. Better combined scores therefore need not mean better component estimates.
All108 diagnostic groups replay exactly. This narrows the next controlled test;
it does not establish a generalization mechanism or overturn the failed risk gate.

- [Conditional-risk diagnosis](outputs/publication_readiness_2026_09/european_easy_component_diagnostic_v1/conclusions.md)
- [Restored fitting-only benefit labels](outputs/publication_readiness_2026_09/european_fitting_gain_labels_v1/results.md)

- [Results and every control](outputs/publication_readiness_2026_09/european_easy_risk_priority_v1/results.md)
- [Decision, interpretation and limits](outputs/publication_readiness_2026_09/european_easy_risk_priority_v1/conclusions.md)
- [Benefit/harm accounting](outputs/publication_readiness_2026_09/european_easy_risk_priority_v1/failure_analysis.md)
- [Locality, seed and tail results](outputs/publication_readiness_2026_09/european_easy_risk_priority_v1/locality_seed_quality.md)
- [Training evidence](outputs/publication_readiness_2026_09/european_easy_risk_priority_v1/training_report.md)
- [Verification scope](outputs/publication_readiness_2026_09/european_easy_risk_priority_v1/verification_report.md)
- [Chinese reproduction guide](outputs/publication_readiness_2026_09/european_easy_risk_priority_v1/operation_zh.md)
- [Prior supervision experiment](outputs/publication_readiness_2026_09/european_easy_hurdle_v1/conclusions.md)
- [Gradient diagnostic](outputs/publication_readiness_2026_09/european_easy_gradient_diagnostic_v1/results.md)

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
