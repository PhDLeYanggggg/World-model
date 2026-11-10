# Songdo: can missing positions change the easy risk decision?

I kept the two edits, the original 2% limit, and the easy cutoff frozen
on the 40 calibration flights. I replayed the already scored 160
confirmation flights across all three seeds. The [calculation](protocol.md)
lets a track enter or leave the easy subset when one or more forecast
steps are missing. It separates a numerical violation from the possibility
that there is no easy reference at all.

| Matched rule | Original three checks / 480 | Also passes all-edited and easy on full grid | Guaranteed selected-easy reference / 480 | Actual releases |
| --- | ---: | ---: | ---: | ---: |
| Independent | 179 | 176 | 229 | 15 |
| Additive | 173 | 170 | 220 | 0 |
| Same family | 71 | 70 | 152 | 0 |
| Joint | 69 | 68 | 144 | 0 |

Every original pass that survived the earlier all-edited check also
survived the full-grid easy check. The selected-easy reference is
guaranteed for 144 joint views versus 229 independent views. For
comparison, the original available-step rule had a reference for 144
and 231 views, respectively. The gap persists when missing steps can
change easy membership. It describes the selected tracks, not a causal
effect of the graph term or a guarantee on new recordings.

All five independent releases that passed the original checks still
pass the full-grid risk sensitivity. The other ten releases and the
zero joint releases do not become deployment successes. The sensitivity
does not change the approved rule or infer physically plausible futures.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_songdo_full_grid_easy_risk.py
```

Run from the original code revision `1b84f8c18` for an exact receipt
replay. The later WEPDTOF audit generalizes the same calculation beyond
two edits and therefore changes the recorded code hash.

`fresh_run`: the threshold-excess and support calculation, using the
same already scored numeric labels. `cached_verified`: the frozen
cutoff, forecasts, matched and actual release counts, and all-edited
readout. `not_run`: new confirmation, training, or deployment.
[Aggregate readout](readout.json) | [Reader supplement v42](../reader_supplement_v2/README.md).
