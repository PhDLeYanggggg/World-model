# Paired Hindcast Replacement on TUMDOT

The [first trial](../tumdot_matched_selector_v1/README.md) recovered
some local passes but paid a large disagreement debit against the
fixed past-motion hindcast. I now directly train expected conservative
utility relative to that hindcast, with the exact masked per-step
target. A retrained control substitutes a CV target and uses the same
network, features, optimizer, samples and mean-square loss. Each
candidate's score is centered at the hindcast score. Replacement
requires positive predicted gain; a guarded decoder additionally
requires no predicted increase in known all/easy positive harm.
The learned guard is not a certificate. The original 2% recording
scorer still determines the actual pass.

All arms keep the exact recipient and peak radius on each row,
including when a selector falls back to the hindcast direction.
There are 4,368 repeated-view edits versus CV across 27 views.
Fitting excludes the held intersection and collection day, with four
donor recordings per held recording and no exact track-ID overlap.
No held outcomes or normalizers enter inference. Three forecaster
and training seeds use nine exposed recordings from one campaign.

| Half radius | Lower vs CV (pp) | Direct lower vs hindcast (pp) | Risk passes / 27 | Joint CV and OLS4 / 27 | Changed from hindcast |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed hindcast | -0.03654 | 0 | 24 | 3 | 0 |
| CV-target control | -0.02446 | -0.11346 | 22 | 6 | 3,023 |
| Direct pair target | -0.02463 | -0.06927 | 24 | 9 | 1,759 |
| Direct pair, guarded | -0.02273 | -0.05710 | 24 | 9 | 1,436 |

The guarded arm improves local joint passes, but not average
conservative utility against either comparator. Its direct observed
contrast against hindcast is +0.01539 pp; the debit is 0.07249 pp.
The conservative contrast by intersection is -0.01004, +0.02876
and -0.19003 pp. The descriptive three-intersection bootstrap is
[-0.19003, +0.02876]. The paired target also loses to the retrained
CV-target control in their **direct** comparison (-0.05749 pp);
two lower bounds against a common third path cannot be subtracted
to reconstruct that contrast.

At quarter radius, all arms pass the original risk gate on all
27 views; the guarded arm still has -0.01051 pp against CV and
-0.02889 pp directly against hindcast. Worst defined signed easy
degradation at half radius is 0.497% for the guard, but it still
fails the full risk/support rule on three views. Native ADE and FDE,
comparisons against both baselines, and per-intersection values are
in [analysis.json](analysis.json).

**Decision:** no policy promotion. The largest loss is concentrated
at one intersection, where observed gain is already negative before
the missing-step debit. It is not explained by that intersection
having the most missing rows. The next experiment should separate
causal action discrimination on complete tails from missing-step
disagreement, keeping the full-roster result primary. More repeated
windows from this same campaign cannot supply independent evidence.
DroneCrowd's 15 calibration and 30 sealed confirmation groups are
unchanged. No new metric/seconds claim, Stage5C or SMC.

```sh
.venv-pytorch/bin/python scripts/run_m3w_tumdot_matched_selector.py
.venv-pytorch/bin/python scripts/run_m3w_tumdot_matched_selector.py --replay
.venv-pytorch/bin/python scripts/run_m3w_tumdot_incumbent_selector.py
.venv-pytorch/bin/python scripts/run_m3w_tumdot_incumbent_selector.py --replay
.venv-pytorch/bin/python scripts/build_m3w_tumdot_selector_replay.py
```

`fresh_run`: 54 new fits and 21,600 updates for the paired-target
trial, 12.332 s, PID 90689, native arm64 Torch with four threads,
zero loader workers. Checkpoints and heartbeat every 100 updates;
rerun to resume. `cached_verified`: matched forecasts, exact
hindcast control and zero-update replay. `not_run`: CREATE or
independent confirmation.
