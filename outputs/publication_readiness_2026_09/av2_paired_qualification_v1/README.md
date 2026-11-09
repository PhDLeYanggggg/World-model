# AV2: Paired Qualification Improves Retrieval, Not Held Release

I separated two questions from the [cost experiment](../av2_forecast_cost_v2/README.md):
whether the fixed independent-agent reference qualifies at all,
and whether a joint choice beats it under the same edit budget.
The earlier experiment found 44 feasible calibration views but
selected a qualifying pair only once.

## Design

Let A denote exact additive eligibility and J denote exact
candidate qualification against that reference. A small model
uses the fixed reference's own causal features to predict A.
Its candidate network has two outputs: an unconditional score
for J and one trained on donor queries where A holds. Both use
weighted classification and same-query positive-negative ranking.
The inputs are candidate geometry, observed-history replay and
relative geometry against the reference. Neither observed
future labels nor availability masks enter inference. An exact
copy of the reference cannot yield strict pair gain, so all
copies of that path are excluded from the new candidate ranks.

I compared three decisions from the same fit:

| Decision | Rank candidates by | Rank scenes for release by |
|---|---|---|
| Direct | Unconditional J | Selected unconditional J |
| Query-weighted | Same unconditional J | A score times selected unconditional J |
| Conditional (registered primary) | J trained where A holds | A score times selected conditional J |

The weighted scores rank examples; they are not calibrated
probabilities or certificates. Query weighting preserves the
direct candidate paths exactly. The original both-arm 2%,
dual-motion-floor and conservative pair checks still decide
whether any nonempty calibration prefix is admissible.
[Selective prediction](https://papers.nips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html)
motivates ranking before abstention, and
[SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html)
learns when to reject. Here both arms must qualify on the same
scenes at the same budget before comparing interaction utility;
I do not borrow a classification-risk guarantee from either work.

The same six cities, 8 observed/12 forecast steps, seeds
17/29/43, 896 added scenarios, candidate bank and frozen outer
additive control are used. The 72 hashed three-city additive
fits and their separately audited cutoffs are reused for donor
supervision. No forecast model was refit. A fold has 245-300
training queries, but only 4-23 eligible reference queries and
2-16 eligible queries with a positive-negative ranking pair.
These are development rotations, not independent confirmation.

## Retrieval and Calibration

| Decision | Calibration top five / 44 | Calibration selected / 44 | Passing prefixes / 1,020 | Held top five / 48 | Held selected / 48 |
|---|---:|---:|---:|---:|---:|
| History, fixed | Not replayed here | 2 | 0 | 12 | 5 |
| Forecast cost, fixed | Not recorded | 1 | 0 | Not recorded | 3 |
| Direct | 11 | 6 | 0 | 14 | 9 |
| Query-weighted | 11 | 6 | 1 | 14 | 9 |
| Conditional | 10 | 6 | 0 | 16 | 8 |

Counts are seed views, not independent samples. On the held
1,020 two-slot views, A has AUROC 0.716 and AP 0.153 at
prevalence 0.062; Brier 0.0596 is worse than the diagnostic
constant-prevalence value of 0.0580. A can help with ordering
without being a calibrated eligibility estimate.

At the highest calibration score in each fold, 12/18 primary
choices fail both arms, four fail the candidate alone, and
two fail the additive arm alone. Where a primary choice does
qualify, the first such choice ranks between 5 and 51 among
scenes in its fold. Candidate positive-harm and additive-risk
checks each fail 11/18 top choices; these checks overlap.
The [failure audit](calibration_failure.json) preserves each
term. The primary has no nonempty release and returns CV.

The query-weighted ablation's single calibration pass comes
from Pittsburgh, seed 29, with Palo Alto held. I scored this
already frozen threshold after the primary readout, without
refitting or choosing a different threshold. It triggers once
on held development data; **neither arm qualifies**. Across
all seeds the fixed additive has 0 eligible Palo Alto views
out of 54 two-slot views. I did not promote the ablation.

## Matched Forecast Results

| Forced policy | Known ADE | Available FDE | Conservative gain vs CV (pp) | Risk passes / 1,932 | Signed easy violations |
|---|---:|---:|---:|---:|---:|
| CV | 0.24668 | 0.57458 | 0 | No edits | No edits |
| Scene uniform | 0.24631 | 0.56954 | -1.24 | 132 | 250 |
| Additive | 0.24574 | 0.56860 | -1.07 | 143 | 242 |
| History | 0.24761 | 0.57486 | -1.60 | 137 | 383 |
| Forecast cost | 0.24602 | 0.57121 | -1.21 | 158 | 346 |
| Direct / query-weighted | 0.24755 | 0.57443 | -1.69 | 134 | 319 |
| Conditional | 0.24664 | 0.57224 | -1.31 | 140 | 243 |

All matched arms make 2,952 edits. The conditional choice
passes both motion floors and risk in 87 edited views versus
77 for additive, but its hard gain is +0.37% versus +0.56% for
additive and its mean scenario p95 ADE is 0.69136 versus
0.68552. Its edited-easy ratio remains undefined in 1,458
views. OLS4 and the raw matched-training EqMotion forecast are
unmatched-budget accuracy controls in the [summary](summary.json).

Direct conditional-minus-additive full-grid pair lower gain
is **-0.006183 [-0.011437, -0.003040]** native units. The
direct trajectory lower gain is **-0.656 [-1.093, -0.243] pp**.
On complete scenarios the pair result is -0.001424
[-0.005799, +0.001701]. Intervals use 3,000 paired city draws
after averaging seeds within scenario. Better retrieval does
not produce better average pair forecasts or a held release.

The next comparison should test causal observability and
easy-support features for reference eligibility. Separate
forecast label support from actual baseline error, and use
past-only actor types and partial observed neighbor tracks
where available. Keep the incumbent comparison and original
checks fixed. Do not add capacity or tune thresholds before
checking which cities have a feasible reference in the menu.

## Reproduce

Requires the audited [forecast cost assets](../av2_forecast_cost_v2/README.md).

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_paired_qualification.py fit
.venv-pytorch/bin/python scripts/run_m3w_av2_paired_qualification.py score
.venv-pytorch/bin/python scripts/audit_m3w_av2_paired_qualification.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2*.py tests/test_m3w_interaction_frontier.py tests/test_m3w_slate_relative_gain.py
```

18 fresh CPU PyTorch fits, native arm64, four compute threads,
zero loader workers. Fit/freeze took 180.05 s (1.72 GB peak
RSS), scoring 18.12 s and audit 32.20 s. Checkpoints every
40 steps retain optimizer and sampler state. Replaying the
completed fit command verified the receipts in 7.30 s.
43 focused tests passed. No CREATE job, manuscript change,
reserved confirmation access, Stage5C, or SMC.

[Registration](registration.json) · [Freeze](freeze.json) ·
[Results](summary.json) · [Failure audit](calibration_failure.json)
