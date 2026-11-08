# Can a Selector Keep the Easy Reference It Needs?

**At equal switch counts, easy-support priority restores some recording
passes, but the new classifier does not beat the simpler mass heuristic.**
Neither policy passes the complete recording requirement. No deployment
or independent confirmation changed.

## Matched Development Experiment

I froze the OpenDD forecast, per-fold uniform blend, previous risk head and
query-local switch counts. I fitted 21 Extra Trees probability heads to
recognize whether a proposed nonidentity edit has observed easy reference
and positive harm at most 2% of that target's CV error. Each head uses
predictions from donors excluded jointly with the evaluation site. Only
past motion and the fixed forecasts are inputs. Until predicted support
accumulates to one in a recording, the new rule prioritizes that event
within the existing per-query risk constraint. The simpler control uses
predicted easy reference mass divided by predicted all reference mass.

Across seven held sites, the event head has AUROC **0.763**, average
precision **0.482** at prevalence **0.183**, and Brier **0.138**. This
demonstrates ranking information, not calibrated recording safety.

| Policy | Flight-seed passes / 606 | Undefined selected easy reference | Affordable easy edits selected | Lower gain vs CV, pp [95% site bootstrap] |
|---|---:|---:|---:|---:|
| Previous query-local | 425 | 75 | 2,932 | +0.289 [+0.180, +0.451] |
| Prioritize estimated easy mass ratio | **453** | **47** | **3,307** | +0.198 [+0.136, +0.287] |
| Prioritize learned affordable-easy event | 444 | 48 | 3,144 | +0.211 [+0.131, +0.331] |

All three rows above have exactly **14,035 switches**, matched within each
query, and the same forecast candidate for each row. The support-first
variant allowed one extra risk-feasible edit at predicted probability
>=0.25 when the local rule would abstain, but made no extra edits on this
roster. It exactly matches the event-priority result; no incremental
effect is claimed.

With the learned rule, 35 previously undefined flight views became defined,
while eight moved in the opposite direction. It gained 46 passes and lost
27 relative to query-local. The simpler mass rule gained 58 and lost 30.
No variant changes the 30 views with no full easy reference. The event
rule gives up **-0.082 pp** [-0.128, -0.041] of conservative gain against
local; against the mass control the difference is **+0.007 pp**
[-0.029, +0.044]. The latter interval includes zero. Intervals resample
seven sites after averaging three seeds (3,000 draws), not individual
overlapping windows or independent cities.

**Decision:** retain the simple mass ratio as the development comparator,
not a deployable controller. The predictable part of easy support is real
enough to reduce undefined views, but does not fix the 2% recording rule
or justify another round of threshold tuning on OpenDD. A useful next
experiment needs a genuinely different past signal or support-rich source,
with source roles fixed before outcomes. Do not open the 30 DroneCrowd
confirmation groups, execute Stage5C, or enable SMC.

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_easy_support.py
```

Requires the frozen velocity and risk checkpoints from the previous two
stages. The local arm64 run took 2.61 seconds and wrote a hash-bound
checkpoint per held site and seed. Independent checkpoint readout matched
byte for byte. No trajectories or weights are distributed.

[Protocol](protocol.md) | [Full metrics](analysis.json) |
[Anonymous working supplement](../evidence_manuscript_v29/README.md)
