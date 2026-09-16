# Cost-Sensitive Regression-Deferral Control

Date: 2026-09-16. Status: implementation and synthetic verification only. Real fitting, development selection, independent confirmation and superiority over this control are **not_run**. The real scientific protocol remains unapproved.

## Literature Mapping

[Mao, Mohri and Zhong, ICML 2024](https://proceedings.mlr.press/v235/mao24d.html) study regression with expert deferral. We inspected sections 2, 4 and 5 of the [authors' arXiv v1](https://arxiv.org/pdf/2403.19494), particularly Eq. (3) and the single-expert reduction. For fixed baseline B and candidate N, let c_B and c_N be their bounded regression costs. Our two-action logistic adaptation minimizes `-c_N log2(p_B) - c_B log2(p_N)`. The source requires bounded losses. This implementation is not a reproduction of its published experiments or a new deferral objective.

## What Is Implemented

`m3w_cost_sensitive_deferral.py` provides linear and one-hidden-layer gates, absolute cost generation, checked out-of-fold supervision, fixed-budget optimization, checkpoint recovery, verified loading and independent routing. `train_m3w_deferral_control.py` is a runnable fit-only entry, not a stub or an automatic test evaluator.

The comparator uses **the same** `risk_features` as the existing gain/harm control, from the same frozen forecaster and causal baseline rollout. Producer checkpoints are loaded through the existing verifier; every held-out training fold is checked against all upstream fitting/selection exposure before label-bearing readers are used. Its targets retain both absolute past-normalized ADE/FDE values, rather than trying to recover them from benefit/harm differences or substituting a one-hot winner.

The approved protocol must explicitly bind `comparators.cost_sensitive_deferral` with `cost_bound`, `width` and fixed `fit_settings`. No real defaults are selected here. Width zero means a linear gate; positive width gives a small GELU network. The fitting seed must be in the protocol. A common positive bound transforms costs as `min(raw_cost / cost_bound, 1)` and the report records clipping separately for each action. The original labels remain in the local OOF cache. Clipped training loss is not untruncated ADE/FDE; comparative evaluation must report the latter separately and use the same approved label eligibility and error unit.

ADE follows the existing cost learner's available-requested-label rule; FDE requires the exact requested endpoint. Ragged requests are explicit prefixes. Missing endpoints do not become the last earlier observation, and future validity does not enter inference features. Formal task/label rules still need reconciliation before real experiments, rather than assuming that training-label eligibility defines the evaluation population.

Each row's total cost weight is retained. Dividing its two weights by their sum without restoring the row weight changes the optimization problem. At an unrestricted optimum the two scores depend on conditional expected costs, not the probability of winning on an individual sample. `deferral_decision` returns a logit margin, not predicted gain or a calibrated harm probability. Ties and unavailable past support use B. Nonfinite logits are rejected. The gate has no fitted easy guarantee, no joint consistency constraint and no physical-safety certificate.

## Recovery And Provenance

The CLI freezes the explicit fold-to-producer map, baseline, protocol, sources, predictor hashes and extraction settings. Complete fold caches have content hashes; missing receipts cause that fold to be regenerated, not silently trusted. Resume rechecks recording caches and ancestor exposure. A group digest binds features, continuous costs and query identities. Duplicate queries, changed costs, changed producers, role mismatch and altered protocol fail closed.

Checkpoint state includes parameters, fit-only feature mean/scale, optimizer, sampler order/cursor, sampler RNG, Torch RNG, step and loss trace. Atomic checkpoints are written at the configured interval; interruptions resume from the last complete update, not a half-applied optimizer step. PID/progress logs are separate for extraction and fitting. Completion emits a `policy` artifact with `family=cost_sensitive_deferral`, parent predictors and fit exposure. Partial checkpoints cannot be loaded as finished policies. Repeating a completed identical run returns `cached_verified` without another update.

CPU and explicit MPS code paths are available with single-process data loading; this slice verifies CPU. It does not claim a new MPS recovery test or a 12-hour endurance result. The existing backend, joint solver, formal evaluation arms and real draft protocol were not modified, avoiding unnecessary invalidation of their checkpoints.

## Verification

See [verification.json](verification.json) for the exact source hashes, executed commands and test results. The cross-process fixture actually fits synthetic forecasters, builds their held-fold predictions, interrupts after a fold and after optimizer step 5, resumes, verifies cache-tamper rejection, and loads the completed artifact. Continuous and resumed linear/MLP fits have identical CPU parameters, losses and sampler states.

The [constructed-cost run](constructed_cost_check.json) trains a linear gate for 300 updates on 100 constant-feature examples. The candidate wins 90% of examples by a small margin but suffers large loss on the remaining 10%. Its constructed expected cost is 0.109 versus the baseline's 0.02. Majority winner classification and row-normalized cost weights choose the candidate; the cost-sensitive optimization keeps the baseline. This is a deliberately constructed mechanism test, **not** real trajectory performance, independent test evidence, or a new scientific result. The softmax value is not a safety probability.

## Remaining Evidence

Follow-up: an [opt-in development comparison](../deferral_development/implementation_and_limits.md) now enforces matched OOF input identity and shared forecasts. The original verification files above retain their original source hashes; the follow-up provides current-version regression evidence. The real development protocol, calibration and confirmation families are still unchanged. Registration must follow approval of capacity, seeds, fitting budgets, clipping sensitivity, labeling and matched coverage/risk rules. The pure feature-level routing function cannot infer whether its caller obeyed data-role boundaries.

This control uses a trajectory-loss adaptation, zero additional deferral charge and finite neural/linear hypotheses. We do not transfer consistency theorems, exchangeability assumptions or a risk certificate to it, much less to constrained multi-agent selection. A larger gain than a threshold baseline will not establish novelty until comparison against this and other appropriate learned controls is completed.

Historical contaminated external numbers remain exploratory. Independent scenes, approved scientific roles and the unchanged CREATE access blocker remain unresolved. No real forecast training, new test result or deployment upgrade was produced in this slice. Coordinates/time remain dataset-local or pixel/raw-frame without unsupported metric/seconds claims. Stage5C and SMC remain off; CVPR readiness remains unachieved.
