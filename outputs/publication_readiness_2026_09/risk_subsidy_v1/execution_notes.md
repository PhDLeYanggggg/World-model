# Execution and Verification Record

2026-09-24. Complete fixed comparison, no new predictor training or deployment.
The [registration](registration.md) and pre-readout repairs were committed at
`889689cdce23f62d3e87472c34c84d76d222e488` before the new decisions/readout.
Routine audits and authorized research steps were performed without another
user sign-off. Independent confirmation remains closed.

## Completed Runs

| Step | PID / session | Result |
|---|---|---|
| Preflight | 89404 / 50024 | Exit 0; 4,498 source bindings checked |
| First decision chunk | 89426 / 92943 | Exit 0; 256 queries, part of full run |
| Remaining decisions with resume | 89602 / 59573 | Exit 0; complete 188,388 query/action/seed instances |
| First outcome readout | 90162 / 86872 | Exit 0; 175,756 rows, all 33 action/policy controls |
| Full decision replay | 90203 / 77005 | Exit 0; exact decision arrays, receipts and manifest |
| Aggregate replay | 90809 / 66800 | Exit 0; exact analysis and outcome-choice archives |
| Separate arithmetic verifier | session 10430 | Exit 0; all implemented checks pass |
| Numerical impact diagnosis | session 49331 | Exit 0; no decision changed |
| Plot and local visual check | completed synchronously | Exit 0; all controls visible, labels inspected |

Runtime: native arm64 `.venv-pytorch`, CPU4 / interop1 / workers0 for the main
passes. No resource probing or multiprocessing. Source models/checkpoints are
cached-verified, not retrained. First-to-last heartbeat intervals for the resumed
decision pass and full replay are 539.969 and 533.672 seconds respectively; these
exclude startup/source hashing and are not total pipeline time or training time.
All required computation processes finished normally. No new CREATE submission
was needed for this fixed local allocation experiment.

## Bound Identities

- Experiment: `bb4ccaf3beeb13f2fc63563ea23bf00c50bedfe844a84ab54ba2ee3b4e187caf`
- Decision manifest: `66d7fcd05d702105d34d90c2a4594308d1558977921f475045bda7c9c693847f`
- Analysis: `06543830cead746151cb3222d9b14b15b5d5371f25a02bcc5ade5500197194a0`

The full identity, raw decisions, local checkpoints, events, and preview PNG
remain under `data/stage_cvpr2027_experiments/`; they are not Git artifacts.
Public artifacts are aggregate tables, receipts, code, configuration, notes and
the generated vector figure. Reproduction requires the locally retained verified
inputs; this is not a self-contained weights/data distribution.

## Checks and Exceptions

[Decision replay](decision_replay.json) and [aggregate replay](aggregate_replay.json)
are exact. [Separate arithmetic](independent_arithmetic.json) verifies:

- 1,318,716 new query/policy risk constraints;
- 1,939 fixed small-query exhaustive optima;
- 1,980 scene/seed/subset reductions and 126 paired contrasts;
- zero exact-count mismatches and zero failed matched-count references.

Three first-query exhaustive checks exceed 16 eligible agents and are explicitly
skipped. Another 119 arm checks have no claimed canonical optimum and are not
counted as verified optima. Across the full experiment, 74,410 query/policy
instances remain canonical-optimum-unverified. This includes two genuine
failed-closed EqMotion instances, not 74,410 failures. All saved new decisions
satisfy the original predicted-risk inequality. Statistical risk is not thereby
certified.

The two failed solves are retained, not repaired after seeing outcomes. Their
supported-ADE primary-score impact is each bounded below 0.000948 percentage
points by a deliberately loose all-query-agent bound. The diagnosis and source
identities are in [numerical_impact.json](numerical_impact.json).

The 63 scoped tests passed before frozen decisions; their unchanged bound version
is reused. No full legacy test-suite pass is claimed. The new descriptive
numerical-impact script completed on the verified real archives; it does not
alter the solver. The earlier independent code review is documented separately
in [code_review.md](code_review.md). The arithmetic verifier is a separate
implementation run by the same agent, not independent scientific replication.

## Research Outcome

The mechanisms affect utility and harm, but the double restriction does not beat
the old strict control. No new best policy, safety guarantee, independent
generalization, neural indispensability, or submission readiness is established.
No external labels or predictions were newly read. Stage5C/SMC remain off; data
remain native annotation pixels/steps without verified metric/time calibration.
See [conclusions](conclusions.md), [all controls](results.md),
[figure](risk_tradeoff.svg), and [recovery instructions](operation_zh.md).
