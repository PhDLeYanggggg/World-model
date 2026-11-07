# Coherent Missing-Label Completions Can Reverse The Pair Ranking

The [previous bound](../native_anchored_pair_completion_v1/README.md)
crosses zero. That alone does not show that both signs are attainable.
This time I constructed explicit image-coordinate completions on the
same 34 exposed recordings. Each preserves every released coordinate
and assigns one position to each missing agent-step across all
overlapping queries, pair edges and three forecaster seeds.

| Completion | Joint minus additive gain (px) | 95% paired source interval (px) |
| --- | ---: | ---: |
| Linear interpolation, constant extension | +0.0064 | [-0.0021, +0.0151] |
| 1x bounded low witness | -0.0808 | [-0.1679, -0.0170] |
| 1x bounded high witness | +0.1043 | [+0.0315, +0.1988] |
| 2x bounded low witness | -0.1033 | [-0.2107, -0.0260] |
| 2x bounded high witness | +0.1272 | [+0.0385, +0.2432] |

Positive means lower pair ADE for joint selection. The two 1x
witnesses are already sufficient to show that the exposed release
does not identify the direction within this coordinate completion
class. Neither is an estimate of the real missing trajectories.
The intervals describe the constructed worlds across 14 sources
with 3,000 paired draws after averaging seeds. They are not
confidence intervals for the underlying unobserved effect.

For each agent, the reference uses linear interpolation between
released positions and constant extension outside them. In each
gap the same bounded two-dimensional direction is used at all
missing times, multiplied by the median released consecutive
displacement (floor 0.25 px) and distance to the nearest release
(cap twelve steps). Both signs use the same radius and fixed
optimizer starts. These are bounds on coordinate perturbations,
not a validated speed model or evidence that an unlabelled
person remained in frame. We do not claim extrema.

The 1,613 original queries are repeated across three seeds.
There are 40,134 distinct missing agent-steps in the full ledger;
31 recordings contain nonzero unknown terms for this contrast.
The released coordinate ledger agrees across history and target
rows. The reference and observed contribution match the original
scorer; each witness is re-scored, preserves released positions,
and falls inside the prior outer bound. No synthetic position
array is exported, and nothing is inserted into the official score.

The forecasts, intervention choices, 2% risk gate, and external
split are unchanged. No independent outcome, new training,
deployment, Stage5C or SMC.

~~~sh
.venv-pytorch/bin/python scripts/audit_m3w_pair_completion_witness.py score
.venv-pytorch/bin/python scripts/audit_m3w_pair_completion_witness.py replay
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_pair_completion_witness.py \
  tests/test_m3w_anchored_pair_completion.py tests/test_m3w_shared_pair_completion.py
~~~

Seven focused tests passed; score and full replay each took about
11 seconds after imports and produced the same aggregate hash.
[Protocol](protocol.md) | [Analysis](analysis.json) | [Replay](replay_execution.json).
