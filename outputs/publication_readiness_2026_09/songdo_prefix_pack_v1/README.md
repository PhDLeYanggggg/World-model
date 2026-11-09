# Songdo: input pack and possible shared sources

I converted the accepted primary-clock fragments from the
[Songdo intake](../songdo_traffic_intake_v1/README.md) into an 8-step
input pack. I left the separately reported drift sensitivity out of
this pack. The purpose is to verify the input and source grouping
before the final external protocol is chosen, not to score a model.

| Site | Query windows | Agent histories | Agent histories with a complete future index |
| --- | ---: | ---: | ---: |
| A | 732 | 28,171 | 24,581 |
| B | 576 | 35,587 | 31,815 |
| C | 30 | 1,319 | 1,219 |
| E | 654 | 15,923 | 13,490 |
| Total | 1,992 | 81,000 | 71,105 |

All 40 session files are represented, including those with no accepted
window. The parser matches each selected history to its original CSV
rows. It reads metadata for the full recording but converts coordinate
values only for the eight observed steps. No interpolated speed or
acceleration field enters the pack. Coordinates are translated by a
single past origin per query, so relative positions are preserved.
CV and OLS4 both produce finite 12-step forecasts on every input;
neither is compared to future coordinates here.

The label index is a separate file holding row pointers, availability
masks and the past coordinate origin. A missing future does not delete
an actor from the input. The export reproduces the intake's pair and
whole-roster counts exactly per accepted fragment. Regression tests
verify that an invalid future coordinate is not parsed, and that
removing one actor's future leaves every input array unchanged.

## A and B should not be treated as independent groups

No exact timestamp-and-position prefix repeats across sites in the
sampled windows. However, the sampled observed-coordinate envelopes of
A and B intersect. I compared their 28,171 and 35,587 prefixes over
overlapping time spans, using a two-unit projected-coordinate tolerance
and requiring four observed time anchors. The diagnostic interpolates
only inside observed spans, not into forecast time. It found no
near-synchronous candidate inside both space and time envelopes.

That result does not exclude vehicles moving between sites at different
times or duplicates outside the sampled prefixes. The conservative
source components for any prospective design are **{A, B}, {C}, {E}**.
They are provisional components within one city campaign, not three
independent test domains and not an assigned train/test split.

## Reproduce

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/build_m3w_songdo_prefix_pack.py
.venv-pytorch/bin/python scripts/audit_m3w_songdo_prefix_linkage.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_songdo_prefix_pack.py tests/test_m3w_songdo_traffic.py
```

The input pack and index files are local under
`data/stage_cvpr2027_experiments/songdo_prefix_pack_v1/`. The exporter
uses one process and writes a hash-bound checkpoint and PID heartbeat
per session. Repeating the command verifies and reuses the existing
pack. `fresh_run`: export, baseline shape checks and linkage audit.
`cached_verified`: publisher archives and source audit. No raw inputs,
coordinates or index arrays in Git. No new split, tuning, scoring,
independent confirmation or change to the original 2% rule.
[Prefix receipt](receipt.json) | [Linkage receipt](linkage.json).
