# Does learning the risk boundary improve action choice?

I took the full and minimum-edit witness plans from fitting sources
and reverified their paths in the common five-amplitude bank. A
training pair keeps every query's edit count fixed. It replaces one
path on the same recipient or exchanges one recipient with a nearby
unedited row in the same query. The alternative must keep positive
conservative gain against both references and defined easy support,
but cross at least one original recording risk inequality.

Two rankers see the same pairs, features, and optimization.
The utility ranker prefers the better worst-reference lower gain;
the risk ranker prefers the action from the feasible plan. Their
bounded scores adjust the same frozen shape model's action gains.
This is a [decision-ranking comparison](https://proceedings.mlr.press/v162/mandi22a.html),
not a new general optimizer. Unlike optimizing a utility loss alone
([SPO](https://pubsonline.informs.org/doi/10.1287/mnsc.2020.3922)),
it isolates the preference induced by crossing a risk constraint.

## Evidence

24 distinct-within-record witness plans across 17 fitting
recordings yielded 390,223 deduplicated pair/label tuples, including
367,393 recipient exchanges. Utility alone preferred the violating
alternative in 68,329 pairs (17.5%). The easy positive-harm
inequality is the most common crossing. The pairs are correlated;
the independent unit used for uncertainty is the source group.

18 native PyTorch fits, 25,200 updates, three source rotations and
three seeds. Held development has 24 recordings from nine source
groups, 72 recording/seed views. The first four policies below
match intervention counts query by query.

| Policy | Edits | Risk passes / 72 | Worst known signed easy change |
|---|---:|---:|---:|
| Frozen shape portfolio | 45,042 | 4 | +10.21% |
| Independent shape score | 45,042 | 3 | +25.85% |
| Utility supervision | 45,042 | 1 | +48.80% |
| Risk-crossing supervision | 45,042 | 1 | +23.59% |
| Utility supervision, free counts | 59,808 | 1 | +5.05% |
| Risk-crossing supervision, free counts | 54,448 | 2 | +5.36% |

| Contrast | Observed-only pp | Completion-lower pp (95% CI) |
|---|---:|---:|
| Risk vs utility, matched counts | -0.063 | -0.590 [-0.958, -0.246] |
| Risk vs independent, matched counts | -0.268 | -0.858 [-1.205, -0.559] |
| Risk vs shape portfolio, matched counts | +0.090 | -0.881 [-1.554, -0.378] |
| Risk vs utility, free counts | +0.047 | -0.575 [-1.249, -0.118] |
| Matched risk vs CV | +0.404 | -0.587 [-1.225, +0.023] |
| Matched risk vs OLS4 | -0.197 | -54.552 [-96.027, -22.595] |

Intervals use 3,000 paired source resamples after averaging seeds.
At matched counts the conservative pair-distance effect is -0.408
local pixels against utility [-0.674, -0.175]. This measures
relative forecast geometry, not physical safety. The calibration
ladder qualifies 0/9 cells. No calibrated intervention is deployed.

**Why the lower worst easy value is not a risk pass.** The matched
utility/risk arms both pass 12/72 on observed data alone, and 11 of
those passes disappear for each arm under the original missingness
treatment. The positive easy-harm test exceeds 2% in 57/58 views,
respectively, with 13 undefined in each arm. Signed easy change
and positive easy harm measure different things. The risk head also
changes 21,414 row actions relative to utility supervision, so the
failure is not due to the two learned rankings being identical.

**Next:** stop turning a recording-level feasibility comparison
into a context-free action score. An otherwise identical edit can
have a different effect depending on the rest of the plan and
remaining risk allowance. Test explicit conditioning on causal
predicted recording gain, risk numerator, and reference support,
with a no-context ranker at matched counts. Do not use observed
future error to set that state.

## Reproduction

```sh
.venv-pytorch/bin/python scripts/train_m3w_cv_risk_crossing.py --replay
.venv-pytorch/bin/python scripts/build_m3w_cv_risk_crossing_replay.py
```

The new fits, scores, and failure decomposition are `fresh_run`.
The mapped training witnesses and shape checkpoints are
`cached_verified`. Remove `--replay` to train or resume with the
local checkpoint lock and heartbeat. The fresh run used arm64,
four Torch threads and no DataLoader workers; it took 128 seconds.
The anonymous archive replays the scalar recording table, not
training or geometry. No new CREATE job.

Eight observed / twelve forecast steps, dataset-local pixels.
No metric, seconds-level, true 3D, foundation or physical safety
claim. DroneCrowd's 15/30 split is unchanged and independent
confirmation is `not_run`. No deployment, Stage5C, SMC or submission.

[Protocol](protocol.md) | [Readout](readout.json) |
[Anonymous replay receipt](replay_receipt.json) |
[Working supplement](../evidence_manuscript_v143/README.md).
