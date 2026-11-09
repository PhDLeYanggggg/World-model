# Does a nearby agent's recent velocity help at the same edit budget?

I compared two ways to add a candidate trajectory. Both start with the
target's eight-step OLS velocity and add a decaying short-term velocity
change. One uses the target's own velocity change; the other uses that
of a nearby agent moving in the same direction. The two arms add the
same number of candidates at the same radii. This tests the value of
the neighbor signal, rather than the value of a larger search menu.
[Frozen protocol](protocol.md) | [Prediction receipt](freeze.json).

## Same slots, different shapes

The reference menu already has OLS4, a median-slope forecast, and a
held-source neural self-history forecast. All three menus keep 3,690
edits, the exact peak displacement slots and every one-edit decision.
I used the same 202 development recordings at three seeds across seven
OpenDD locations. Each new arm adds 6,223 row-slot candidates across
200 of 439 edited recording-seed views. The direct replacement changes
the shape at the same 1,038 recipients in both arms, without moving an
edit to a different agent.

| Menu | Full-grid positive, ignoring risk and individual accuracy | Full-grid positive under all constraints | Distinct recordings under all constraints |
| --- | ---: | ---: | ---: |
| Original expanded menu | 233 / 439 | 21 / 439 | 13 |
| Plus own velocity change | 235 / 439 | 26 / 439 | 16 |
| Plus neighbor velocity change | 236 / 439 | 27 / 439 | 17 |

The constrained solution must pass the original three 2% risk checks,
beat CV and OLS4 on conservative individual accuracy, and not lose
individual accuracy to the original additive selector. It also needs a
positive gain in full-grid relative-displacement error, including the
missing-step debit. I recomputed both objective variants wherever an
extra candidate could fill an existing slot. All constrained solver calls
returned optimal or infeasible; no timeout was counted as infeasible.
These counts describe what is possible with future labels, not what the
causal replacement rule delivers.

## Without hindsight

At the same recipients and radii, the own-history rule satisfies risk
and all accuracy constraints on 31/439 edited views; the neighbor rule
does so on 33/439. They also improve relative motion on 6 and 7 views,
respectively. The rest are not confirmed gains. The direct peer-minus-own
accuracy difference is **+0.00062 pp** of CV error, 95% location bootstrap
**[-0.00749, +0.01036]**. The full-grid relative-displacement gain is
**+0.00000755** native coordinate units, 95% interval
**[-0.00006589, +0.00009128]**. Positive is better. These are averages
over six estimable locations; the seventh has no edited comparison.
Only three locations have direct replacements. The point estimates are
small, the intervals cross zero, and the largest positive location
effect is not shared by the other two locations with replacements.

**Decision:** Extra forecast shapes recover a few verified opportunities,
but substituting this neighbor's recent velocity for the target's own
history does not give a reliable advantage. I would not scale this rule
or train a larger pair head on it. The next candidate needs a stronger
source of interaction information than proximity and heading agreement,
with a comparison against the same self-history bank and an independently
labelled source. [JFP](https://proceedings.mlr.press/v205/luo23a.html)
learns pairwise potentials for joint future prediction, and
[GroupNet](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_GroupNet_Multiscale_Hypergraph_Neural_Networks_for_Trajectory_Prediction_With_Relational_Reasoning_CVPR_2022_paper.html)
learns group structure. This experiment does neither: it tests whether
a specific causal neighbor signal adds value under fixed slots and
risk, not whether relational forecasting in general is useful.

## Replay

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_peer_transient.py freeze
.venv-pytorch/bin/python scripts/run_m3w_opendd_peer_transient.py score
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_peer_transient.py tests/test_m3w_interaction_frontier.py tests/test_m3w_opendd_pair_attribution.py
```

`fresh_run`: candidate inference and scoring. `cached_verified`: source
cohorts, budgets, donor neural weights and parent score receipts. Local
arm64 CPU, four compute threads, no worker processes, no new training.
Each fold has a hash-bound checkpoint and PID heartbeat. The initial
score stopped when aggregation supplied a recording key twice on reused
rows; the repair preserves the frozen candidates and resumes from the
first three completed fold checkpoints. The report verifies all 21.
The bootstrap uses 3,000 location draws and overlapping donor folds, so
it is descriptive. No reserved data, calibration thresholds, manuscript,
deployment, Stage5C, SMC, metric or seconds claims changed.
[Structured result](readout.json).
