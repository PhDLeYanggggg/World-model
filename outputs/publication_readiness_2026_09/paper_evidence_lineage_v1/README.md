# Source roles behind the forecast replacement results

I checked the local cohort indices, artifact hashes and declared fitting
roles behind the main results. The check does not reopen external outcomes.
Its scope is the listed frozen experiments, not every historical M3W run.

| Evidence | What I verified | How it is used |
| --- | --- | --- |
| OpenDD count comparison | 202 distinct recording IDs, 1,819 distinct publisher object IDs across seven locations, and 3,395 nonoverlapping queries within recordings. The 63 inner rankers exclude donor and outer held sites, the 42 count heads use donors only, and the 21 outer rankers omit their held location. The hindsight and online readouts bind the same registered setup. | Exposed development. 606 seed views are not 606 independent recordings. |
| Moving displacement bound | Nine frozen forecast banks over three folds and three seeds. Each fold separates three fit, three calibration and three held sources within the same nine-source roster. All 24 scored recordings have all three seeds. | Exposed development for a hindsight bound. This is the smaller bound cohort, not a certification of every Moving experiment. |
| WEPDTOF | The approved roles, training and pilot freezes, both forecast freezes and external readout agree. One development source is excluded from the 13 external sources and 15 clips. Three donor-only checkpoints and 30 saved forecast packets match their hashes. | Frozen external evaluation, 45 repeated seed views. Not a positive release result. |
| Songdo | 40 calibration flights in three components and 160 confirmation flights in 14 different components. Git contains the calibration receipt before the confirmation receipt; hashes and frozen thresholds match the local records. | One-shot confirmation after calibration within one acquisition day. Not city-to-city transfer. |
| DroneCrowd | The calibration gate is false and no confirmation readout is present in the registered output location. The audit does not access the 72 reserved recordings. | Calibration only. |

I also reran OpenDD with the archived PyTorch checkpoints on native arm64.
All 21 held-location/seed cells finished in 16.984 seconds of runner time
with zero optimizer updates. The runner compared its recomputed score and
freeze to the original files. This replay reads existing development targets;
the lineage audit reads OpenDD index arrays, file hashes and score receipts
instead. Neither action is new training or new independent confirmation.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_paper_evidence_lineage.py
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/run_m3w_recording_count.py --replay
```

The [aggregate receipt](receipt.json) contains no recording IDs, checkpoints
or trajectories. See the [Chinese guide](../../../reproducibility/M3W_PAPER_REPLAY_ZH.md)
for the environment, offline package and checkpoint recovery steps.
