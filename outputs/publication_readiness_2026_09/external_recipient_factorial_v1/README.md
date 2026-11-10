# Does joint ranking change the route or the recipient?

The frozen Songdo test suggested that different recipients account
for much of the contrast between joint and independent selection.
I wanted to test that explanation prospectively, rather than choose
new contrasts after seeing another dataset's targets. On the
[Athens pNEUMA pilot](https://open-traffic.epfl.ch/) and the
[Swiss roundabout release](https://zenodo.org/records/15077435), I
crossed the independent, additive, and joint recipient sets with
all three path rules. Each cell uses the same candidate bank, two
distinct edited vehicles, and the original peak displacement.

## What changes before seeing the future

| Cohort | Matched query-seed views | Joint vs additive changes recipients | Joint changes paths on additive recipients | Joint changes paths on joint recipients |
| --- | ---: | ---: | ---: | ---: |
| Athens | 1,929 | 1,705 | 233 | 100 |
| Swiss | 7,431 | 4,650 | 1,343 | 1,107 |

| Cohort | Original joint vs additive ADE envelope | Fixed additive recipients | Fixed joint recipients |
| --- | ---: | ---: | ---: |
| Athens | 0.006222 | 0.000356 (5.7%) | 0.000129 (2.1%) |
| Swiss | 0.025343 | 0.004834 (19.1%) | 0.003148 (12.4%) |

For each query, the average distance between two forecasts bounds the
absolute difference in their ADE against any common full set of
future positions. I use the entire query roster, including agents
that neither rule edits. The percentages in parentheses compare
those envelopes within the same cohort. The drop holds separately
on all four Athens dates and all three Swiss sites. It does not show
which rule is more accurate, and the two pilot coordinate planes
are not pooled or treated as validated metric distances. FDE and
relative-path envelopes are in the receipts.

All three original diagonals replay. An initial shortcut rebatched
the selected pairs, which changed a near-tied path family in Athens.
The final implementation uses the original scoring block shapes
and tie order, and discards unused rows only after inference.
I kept all 258 unmatched Swiss views in the full-roster risk check
at CV, without counting them as matched comparisons or risk passes.

## Reproduction

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/freeze_m3w_external_recipient_factorial.py athens
.venv-pytorch/bin/python scripts/freeze_m3w_external_recipient_factorial.py swiss
.venv-pytorch/bin/python scripts/rehearse_m3w_external_recipient_factorial.py
```

`fresh_run`: 72 local file-seed packets; fixed-recipient inference and
geometric bounds. `cached_verified`: all original causal inputs and
three-seed OpenDD ensembles; no refitting. The [readout rehearsal](rehearsal.json)
uses fabricated targets on the full frozen packets to check masks,
matched support and the unchanged recording risk rule. It exports no
accuracy numbers from those targets. `not_run`: formal Athens and
Swiss outcome scoring. Date/site roles and the release evaluation
must be settled before any outcome access. Athens spans four dates
in one city; the three Swiss sites were recorded on one day.

[Protocol](protocol.md) | [Athens freeze](athens_freeze.json) |
[Swiss freeze](swiss_freeze.json). No new deployment decision,
CREATE job, Stage5C execution, SMC or OpenReview submission.
