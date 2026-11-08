# Matching the Decision Budget Does Not Rescue the Ranker

**Result: no established gain over CV or over pointwise gain regression.**
This is development on the existing OpenDD cohort, not an independent
confirmation. No thresholds were fitted or swept. No deployment changed.

I trained 63 fresh Torch models: three objectives, three seeds, and seven
held locations. All arms share the 10,475 targets, 3,395 past-only queries,
fixed forecasts and training draws. The pointwise model predicts gain;
the query model adds a regret loss over all legal edits; the dual model
adds conservative OLS4 harm to that ranking loss. The primary comparison
fixes one edit per supported query. A second comparison fixes the same
approximately 20% count per site, with each arm ranking its own queries.

| Comparison | Paired lower gain, pp [95% site bootstrap] |
|---|---:|
| CV ranking vs pointwise, one edit | +0.054 [-1.409, +1.578] |
| Dual ranking vs pointwise, one edit | -1.086 [-1.956, -0.137] |
| CV ranking vs unchanged CV, one edit | -1.051 [-2.339, +0.226] |
| Dual ranking vs unchanged CV, 20% budget | -0.012 [-0.129, +0.088] |
| CV ranking vs pointwise, fixed per-agent candidate | -0.212 [-0.388, -0.073] |
| CV ranking vs pointwise, fixed recipient | -0.350 [-1.997, +0.867] |

There were 10,122 edits at full coverage and 2,016 at 20%, summed over
three seeds. The dual ranker at 20% had worst signed descriptive easy
degradation of +1.73%; it did not improve overall accuracy over CV.
At full coverage the CV ranker had observed ADE/FDE 0.6122/1.3594 versus
CV's 0.6083/1.3534 in publisher coordinates. Missing steps remain in the
conservative comparison, with no change to the official easy-risk rule.

The offline, outcome-aware oracle gained +16.805 pp over CV with the full
menu, +6.392 pp when it could only choose the recipient and +10.110 pp
when it could only choose the candidate. Those restrictions use a
history-hindcast control, so the two numbers are not additive and are not
upper bounds for all possible restrictions. Both decisions leave room
in hindsight. The learned ranking did not improve either at matched
counts; this loss did not recover the gains from these inputs. At 20%,
the hindsight comparison still finds +8.386 pp. It is
not a deployable selector or a safety guarantee.

**Decision:** keep the current deployment. More threshold search or
another loss on this same causal feature set is not the immediate priority.
Test a candidate forecast or an independent past-only signal that can
change the attainable accuracy, before repeating the intervention study.
The 30 DroneCrowd confirmation groups remain sealed; no Stage5C or SMC.
These results are included as negative evidence in the
[anonymous working supplement](../evidence_manuscript_v27/README.md).

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_location.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_query_rank.py
```

The run took 40.91 seconds on native arm64 CPU with four compute threads
and no workers. Checkpoints every 100 updates, site-bound data and a PID
heartbeat allow recovery; re-running from checkpoints matched the result
byte for byte in 1.94 seconds. No source tracks or checkpoints are shared.

[Protocol](protocol.md) | [Full per-site results and losses](analysis.json) |
[Fresh run](execution.json) | [Exact checkpoint readout replay](replay_execution.json)
