# Songdo Traffic: paired labels for an external test

I audited a new source outside the existing OpenDD development sites and
without using the sealed DroneCrowd groups. [Songdo Traffic v2](https://zenodo.org/records/17924857)
is a CC BY 4.0 release from a Korean multi-drone campaign. I took the first
published date (October 4, 2022) and the first four available intersection
IDs in alphabetical order (A, B, C, E), not a selection by forecast scores.
I downloaded 465,687,575 bytes across four ZIPs, matched their publisher
MD5s and recorded SHA-256 hashes. The audit covers all 40 session CSVs,
27,778,328 observations, and 27,027 file-local vehicle IDs. It does not
count those IDs as unique people or vehicles across the campaign.

## What can be evaluated?

The structural audit reads only IDs, visibility, class and timestamps.
It does not parse coordinate columns or calculate forecast error. I
separate session, drone and contiguous time fragment. I do not join
across gaps over one second or interpolate missing frames. Within each
fragment I reconstruct integer steps from successive distinct timestamp
gaps and verify both local rounding error and accumulated clock phase.
The primary check accepts **116 of 141 fragments** (21,989,584 rows).
The other 25 fragments stay out of the primary counts.

The publisher documents a nominal camera rate of 29.97 Hz; for accepted
primary fragments a stride of 12 nominal frames gives approximately
0.4004 s between retained samples. This is specific to this release and
does not establish a time or metric scale for SDD. The released data include
planar coordinates in EPSG:5186. The [source paper](https://doi.org/10.1016/j.trc.2025.105205)
distinguishes un-interpolated positions from smoothed and interpolated
kinematic estimates. Those speed and acceleration columns must not be
used as causal features. Released tracks are not independent human gold
or proof of an online camera tracker.

The table uses only publisher-flagged fully visible positions for the
eight observed samples and twelve forecast samples. A query enters the
roster based on observed support, not future labels. I take the earliest
available multi-agent query, then skip forward past its full 20-step span.
A missing track is kept in the denominator for that selected query.

| Site | Accepted / all fragments | Nonoverlapping multi-agent queries | At least one pair with full future | Full future for the entire observed roster |
| --- | ---: | ---: | ---: | ---: |
| A | 38 / 40 | 732 | 732 | 46 |
| B | 29 / 35 | 576 | 576 | 12 |
| C | 2 / 18 | 30 | 30 | 8 |
| E | 47 / 48 | 654 | 653 | 58 |
| Total | 116 / 141 | 1,992 | 1,991 | 124 |

There are 1,571,153 fully observed future pairs out of 1,990,293 pairs
in these 1,992 queries (78.94%). A complete pair does **not** make the
entire scene scorable without missing labels. There are only 124
complete rosters (6.22%). Neither queries nor pairs are independent
statistical samples. Counts under the any-visibility alternative are
included in the readout, not substituted into the primary comparison.

## Clock sensitivity: C is not short of data

I separately allowed a small affine clock drift without changing a
single inferred integer index or bridging any gap. The fitted period
must stay within 0.1% of the nominal period, with the same local gap
requirement and a bounded post-fit residual. All 16 originally excluded
C fragments pass this check, with fitted periods 33.36340 to 33.36345 ms.
This recovers 3,785,064 rows and adds 458 strict nonoverlapping queries
with a complete pair, 106 with a complete roster. The nine excluded
fragments elsewhere are not rescued. The original counts are unchanged;
this is an explicit sensitivity result, not a silent eligibility change.

## Source decision

None of the audited time fragments overlaps across different sites on
the same recorded drone, and there are no reused IDs across drones within
a session. Neither observation proves that tracks cannot link between
intersections. Songdo is **one new city and one acquisition campaign**,
not four independent domains. Before any formal scoring, fix the policy,
the clock rule, the source grouping, and whether this is zero-shot or
uses separate calibration groups. My recommendation is a one-time
zero-shot test without fitting on Songdo, with the existing 2% rule and
both complete-pair and full-grid scores. Do not select the test roster
on complete futures or infer safety from label coverage.

## Reproduce

```sh
dir=external_data/Songdo_Traffic_source_audit
mkdir -p "$dir"
for site in A B C E; do
  file="2022-10-04_${site}.zip"
  if test ! -f "$dir/$file"; then
    curl -fL --continue-at - --output "$dir/$file.partial" \
      "https://zenodo.org/api/records/17924857/files/$file/content" &&
      mv "$dir/$file.partial" "$dir/$file"
  fi
done
.venv-pytorch/bin/python scripts/audit_m3w_songdo_traffic.py
.venv-pytorch/bin/python scripts/audit_m3w_songdo_traffic.py --clock-sensitivity \
  outputs/publication_readiness_2026_09/songdo_traffic_intake_v1/analysis.json
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_songdo_traffic.py
```

Raw trajectories stay outside Git. This was a local CPU intake, not a
training run, policy comparison, independent confirmation, new split,
or change to the original risk rule. [Primary receipt](analysis.json) |
[Clock sensitivity](clock_sensitivity.json).
