# Better Selected-Harm Prediction Did Not Improve The Matched Policy

I trained two matched Torch heads on the candidates that a source-excluded
version of the original filter would consider. Both predict known easy
positive harm. One learns its expected magnitude directly; the other also
learns occurrence and severity separately. I held all other costs and the
original count of substitutions in every query fixed. This tests a model
change, not another threshold adjustment.

| Ranking | Fixed raw-choice harm-fraction MSE | Switches | Original passes / 102 | Easy over 2% |
|---|---:|---:|---:|---:|
| Original | 0.018657 | 6,184 | 2 | 64 |
| Easy excess with ridge harm | 0.018657 | 6,184 | 2 | 64 |
| Easy excess with direct head | 0.017256 | 6,184 | 2 | 64 |
| Easy excess with factorized head | 0.017310 | 6,184 | 2 | 64 |

Relative to the direct head, the factorized head changes the paired
completion-lower utility by **-0.000774 pp** (95% source-bootstrap interval
[-0.002810, +0.000706]). It reduces known easy positive harm by +0.000122 pp
[-0.000202, +0.000458] on the full known CV reference scale, not the
original selected-risk scale. Neither contrast supports a useful change.
Both new heads improve MSE relative to the raw ridge estimate on the same
6,184 original choices, but improve neither original pass counts nor
utility over the raw ranking. All four arms retain 36 undefined easy
views. The raw choices and scoring reproduce exactly. The 3,000 paired
bootstrap draws resample 14 sources after averaging the three seeds.
The factorized head changes 328 row decisions relative to the direct head;
the lack of downstream gain is not an identical-action artifact.

## Can A Perfect Ranker Solve The Screened Problem?

I gave the offline solver the future labels and used the same original
risk, easy reference and signed net tests. No oracle choices are saved.

| Oracle action space | Verified positive feasible / 102 | Infeasible | Unresolved |
|---|---:|---:|---:|
| Screened pool, exact raw count per query | 30 | 72 | 0 |
| Full bank, exact same counts | 72 | 20 | 10 |
| Screened pool, counts free up to the original cap | 57 | 45 | 0 |

The screened matched case has 26 views with no eligible easy reference at
a positive-count query, 25 impossible by the all-excess lower bound, 11
impossible by the easy-excess lower bound, and 10 with no switches. The 10
unresolved full-bank solves reach the fixed time limit without a witness;
72 is a verified lower bound on feasibility, not a claim of exact full-bank
coverage. All reported witnesses pass the original evaluator. The previous
102/102 unrestricted free-count oracle is a different action space.

**Decision.** Do not continue tuning the small-pool ranker. The model can
still miss feasible solutions, but a perfect ranker cannot recover the
72 infeasible screened views. Test an allocation rule that keeps alternatives
available through the recording-level decision and controls the total harm
and reference together, rather than a pointwise gate that deletes useful
support. The offline diagnosis on exposed A cannot replace independent
calibration and confirmation.

This is a conditional risk-estimation experiment, not novelty for basic
factorization or regression with deferral. [Vasilyev et al.'s conditional
risk preprint](https://arxiv.org/abs/2604.20409) treats expected loss
estimation as regression; [Mao et al.](https://proceedings.mlr.press/v235/mao24d.html)
give regression with multiple-expert deferral. The question here is whether
a better estimated selected cost changes the downstream result under fixed
counts and the original recording-level positive-harm and support rules.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_selected_positive_harm.py prefix
.venv-pytorch/bin/python scripts/run_m3w_selected_positive_harm.py fit
.venv-pytorch/bin/python scripts/run_m3w_selected_positive_harm.py score
.venv-pytorch/bin/python scripts/run_m3w_selected_positive_harm.py replay
.venv-pytorch/bin/python scripts/run_m3w_admissible_pool_ceiling.py score
.venv-pytorch/bin/python scripts/run_m3w_admissible_pool_ceiling.py replay
```

The 20-step prefix resumes. The first fit later stopped at a raw-control
argument omission, after saving two 400-step models. Revision `fddd93fe`
fixes the call only. Both completed models are reused after matching the
head code, training tensors and sampler hashes; the remaining 82 models
are fresh. Total 84 native Torch fits and 33,600 actual optimizer updates.
Per-head donor pools have 8,387 to 24,092 candidates, with 1,775 to 5,240
positive events. Final fit process 54.11 s, 2.91 GB peak RSS; CPU4,
interop1, workers0. The scores replay exactly. All paths from the same
three seed banks and all 102 repeated A views stay fixed. No independent
confirmation, deployment, rule change, Stage5C or SMC.

The oracle runs replay all statuses and scalar outcomes exactly
(109.74 and 111.33 s), including the ten unresolved views.

[Protocol](protocol.md) | [Scalar results](analysis.json) | [Replay](replay_execution.json)
| [Oracle results](oracle/analysis.json)
