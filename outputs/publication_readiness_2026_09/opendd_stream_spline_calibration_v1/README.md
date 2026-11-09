# Calibrating on the interventions the controller actually makes

I tested whether the spline selector failed because its calibration
examples did not match its deployment inputs. I kept the same outer
checkpoints, candidates, recipients, query times and displacement peaks.
For calibration I replaced the broad, history-based proposal roster
with slots from the chronological controller itself. The calibration
count head and its ranker excluded both the outer location and the
calibration location. Rankers used to make count-training menus also
excluded the source of each training menu.

I trained 63 count heads for 44,100 optimizer updates, using three
seeds on the native arm64 environment. I reused the 168 rank and 84
shape-selector checkpoints after role and hash checks. Across the
overlapping calibration folds, actual slots numbered **8,715**, versus
**117,954** proxy slots. These are not independent observations.

| Shape rule at the same 1,457 outer slots | Spline substitutions | Risk passes | Risk and both accuracy references |
| --- | ---: | ---: | ---: |
| Analytic anchor | 0 | 112 | 77 |
| Past-prefix control | Not applicable | 124 | 90 |
| Proxy-calibrated joint | 4 | 112 | 77 |
| Streaming gain only | 2 | 112 | 77 |
| Streaming independent risk | 0 | 112 | 77 |
| Streaming joint risk | 0 | 112 | 77 |
| Streaming joint, calibrated with original recording scores | 0 | 112 | 77 |

Streaming joint vs proxy joint is **+0.010 conservative pp** (95%
paired location bootstrap **[0.000, +0.024]**, 3,000 draws after seed
averaging). It removes four unhelpful spline substitutions but adds no
passing view. Against past prefix it gives **+0.019 pp [-0.013, +0.066]**
and passes fewer risk checks. All four positive-gain joint grid entries
on streamed calibration slots violate easy excess. The exact-recording
calibration selects fallback on all seven folds as well. Zero spline
substitutions leaves the analytic edits in place, not unmodified CV.
Worst signed easy degradation stays **11.28%**; 334/606 views have
undefined selected easy support and do not pass.

## How much can a same-query decision change?

Only **55/1,402 edited queries** contain two slots, across 49 of the
606 recording-seed views. A label-aware search over the original and
spline paths, confined to those double-slot queries, can qualify
**80** views under the exact rule, versus **77** for the unchanged
anchor. The same search on all slots recovers the existing capacity of
**170**. No solve is unresolved. This is a limit of this shape bank and
fixed schedule, not a limit on interaction-aware forecasting in general.

The next comparison should move the coupled decision earlier: use
current and past observations to propose support-bearing recipients or
to allocate scarce edits across successive queries, then compare at the
same realized intervention budgets. Another penalty on the same two-slot
shape problem cannot address the much larger single-slot bottleneck.
This development result does not change deployment, establish physical
safety, or use the 30 independent DroneCrowd groups. Stage5C and SMC remain off.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_opendd_stream_spline_calibration.py --replay
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/audit_m3w_stream_spline_joint_opportunity.py
```

Omit `--replay` to train or resume missing count checkpoints. The first
command reconstructs the same streaming calibration and outer forecasts
before scoring; the second uses labels only for the explicit hindsight
capacity check. This round ran locally, not on CREATE.

[Protocol](protocol.md) | [Freeze](freeze.json) | [Readout](readout.json) |
[Opportunity diagnostic](joint_opportunity.json) |
[Manuscript v151](../evidence_manuscript_v151/README.md)
