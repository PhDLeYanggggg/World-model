# Conditional Risk Supervision at Matched Recording Budgets

**Fresh Torch run:** 54 heads, nine held-source folds, three actual
seeds and two paired objectives. Each head makes 500 AdamW updates
(27,000 total). Initialization, sampled rows, architecture, causal
features and the ten candidate paths match within each pair.
Conditional supervision adds label-availability, easy-event,
conditional reference-mass and conditional effect losses to the
same marginal losses. The training easy cutoff uses only the eight
fitting groups; original evaluation cutoffs and the 2% rule stay
fixed. No oracle choices, calibration or confirmation labels train
the heads.

The fixed price grid ranks candidate actions at the original
recording totals. The comparator ranks predicted gain alone.
A query-count control keeps the same total and also preserves the
original distribution across queries. These are forced-count
diagnostics: if no predicted feasible choice is found, the scorer
keeps the least predicted excess at the same budget. Such a
replacement is not a deployment decision.

| Allocation | Conservative gain vs CV (pp) | All positive harm >2% | Original risk passes |
|---|---:|---:|---:|
| Marginal gain only | -3.438 | 63/72 | 0/72 |
| Marginal, risk ranked per recording | +1.494 | 44/72 | 0/72 |
| Conditional gain only | -9.229 | 66/72 | 0/72 |
| Conditional, risk ranked per recording | +1.140 | 45/72 | 0/72 |
| Conditional, risk ranked at fixed query counts | +1.334 | 58/72 | 0/72 |

Every arm makes **95,769 changes** across 24 recordings and three
seeds. Seeds are repeats of the same data, not independent scenes.
Primary direct conservative contrast, conditional versus marginal
recording allocation: **-0.415 pp [-0.795,-0.081]**. Against full
OLS4 the conditional record arm is **-10.123 pp [-18.415,-1.543]**.
Intervals use 3,000 paired source-group resamples after averaging
seeds. Conditional recording vs conditional query is -1.530 pp
[-3.679,+0.385]: fewer violations do not establish utility lift.

## Why the risk gate still fails

Selected easy reference mass is overestimated in 46/72 views with
conditional supervision, versus 61/72 with marginal-only training.
Only 57/72 views have both actual denominators for either record arm.
On these fixed choices, using actual reference with estimated harm
gives 0/72 two-ratio passes for both arms. Using actual harm with
estimated reference gives 12/72 for marginal and 5/72 for conditional,
but some are undefined under actual support. These component swaps
are diagnostics, not policies or calibrated certificates.

We next freeze each head's recipients and replace only the paths
using fitting outcomes. On known labels, minimum ADE minimizes both
positive-harm numerators and signed easy harm; on wholly missing
labels, minimum maximum disagreement minimizes the debit. Because
reference masses are fixed, this is an exact risk feasibility check
within the nonidentity bank. **All three checked recipient sets
remain 0/72**, even with hindsight paths. This does not optimize
dual-reference utility. The [prior allocation audit](../recipient_release_capacity_v1/README.md)
finds 14/24 hindsight witnesses at the same recording totals when
recipients may change. These two results isolate an actual
recipient-selection failure on this bank.

This is distinct from [selective regression's subgroup coverage
criterion](https://proceedings.mlr.press/v162/shah22a.html): the present
test holds intervention counts fixed and scores selected reference
and harm in each recording. Do not promote this head. The next
test should learn recipient feasibility at matched count, rather
than refit another path ranker on these same recipients.

~~~sh
.venv-pytorch/bin/python scripts/run_m3w_matched_support_moments.py
.venv-pytorch/bin/python scripts/run_m3w_matched_support_moments.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_matched_support_risk.py
.venv-pytorch/bin/python scripts/audit_m3w_matched_support_recipients.py
~~~

Native arm64 CPU, four compute threads, one inter-op thread, zero
loader workers. Checkpoints and heartbeat every 100 updates; rerunning
resumes the same fold and RNG state. All held choices freeze before
scoring. Exact score replay does not retrain. Nineteen scoped tests
pass. Checkpoints and selected rows remain local. No new deployment,
Stage5C, SMC, metric/seconds claim or independent confirmation.
[Analysis](analysis.json) | [Risk components](mass_audit.json) |
[Recipient check](recipient_audit.json).
