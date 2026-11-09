# P-DESTRE: usable video access, but one linked source

I audited P-DESTRE as a potential external source after the fixed
two-edit DroneCrowd policy failed calibration. The video is accessible
without downloading the full archive. The annotations support many
8-observed/12-predicted windows, but the videos are not independent
participant groups.

## What I verified

The [publisher release](https://p-destre.di.ubi.pt/download.html)
contains volunteer recordings from two university campuses and
provides the video and frame-level annotations. The
[dataset paper](https://www.di.ubi.pt/~hugomcp/doc/PDESTRE.pdf)
documents IDs that persist across recording sessions. These are
released reference tracks, not tracks from our own online detector.

I scanned the uncompressed archive with exact HTTP byte ranges
and a pinned ETag and size (41,535,221,248 bytes). The index
found 75 matching video/annotation pairs after discarding hidden
AppleDouble metadata. The structural audit retrieved the 126.06 MB
of annotation members and parsed only frame and anonymous release ID
columns. It did not parse boxes, pose, demographics or appearance,
and does not publish individual IDs. There are 1,280,739 rows with
known IDs and 196,646 rows with unknown ID, which cannot be stitched
into identified tracks. The known-ID rows contain 261 distinct
release IDs with no duplicate frame-ID pairs within a clip.

Separately I streamed one actual MP4 through a capped local reader
and decoded its first eight frames without saving any images. The
first successful run requested 11.53 MB of compressed bytes and
returned a 3,840 x 2,160 raster. The container reports 1,347 frames
at 29.97003 fps; the clip's last known-ID frame is 1,327. This checks
one video's accessibility and reported clock, not corpus-wide timing,
frame-to-label alignment or any time/geometry claim for SDD.

## Label support

Selection is greedy in time, using two or more exact eight-step
histories and the annotated extent as a boundary. It does not
require future labels for any selected person. At each stride, the
full 20-step spans do not overlap within a recording. Stride 12
is a structural sensitivity check, not a proposed protocol change.

| Audit grid | Multi-agent queries | History targets | Complete 12-step targets | Partial / absent futures | Queries with 2+ complete targets | Whole roster complete |
|---|---:|---:|---:|---:|---:|---:|
| Native frame indices | 5,113 | 59,323 | 54,354 (91.6%) | 4,776 / 193 | 5,070 | 2,228 |
| Stride 12 | 417 | 3,686 | 2,191 (59.4%) | 1,433 / 62 | 355 | 45 |

Two complete tracks do not imply that a whole online query can be
scored. These counts use released IDs and say nothing about whether
an online detector can recover the same roster. Neither the windows
nor the 75 videos are independent statistical units.

## Independent evidence decision

Connect two videos if any known release ID occurs in both. **All
75 videos form one connected component**, spanning 11 date prefixes.
A random video or date split can put the same identity on both
sides. I will not use this corpus for simultaneous participant-disjoint
calibration and confirmation. It may still serve as **one** fully
external source for a frozen transfer or online-support stress test,
if the model and detector training lineage are disjoint. It cannot
supply multiple independent clusters for a source-level interval.

The [DroneCrowd perfect-switch bound](../dronecrowd_perfect_switch_ceiling_v1/README.md)
already shows why re-tuning the original two-edit release cannot
close its all-agent OLS4 gap. The locked 15/30 role manifest was
verified again, but the 30 confirmation groups were not opened.
P-DESTRE adds a real raw-video route, not a substitute for that
confirmation or for the paper's missing independent comparison.
Keep it out of official scoring until the whole-source role and
training provenance are fixed. Prioritize a cohort with genuinely
separate acquisition groups for the main comparison.

## Reproduce

```bash
export SSL_CERT_FILE=/etc/ssl/cert.pem OPENBLAS_NUM_THREADS=1
.venv-pytorch/bin/python scripts/audit_m3w_pdestre.py index
.venv-pytorch/bin/python scripts/audit_m3w_pdestre.py audit
.venv-pytorch/bin/python scripts/probe_m3w_pdestre_video.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_pdestre.py tests/test_m3w_external_causal_sources.py tests/test_m3w_topdown_source_support.py tests/test_m3w_kit_ais.py
```

`fresh_run`: archive index, annotation support and video decode.
`cached_verified`: local per-member audit replay and unchanged
DroneCrowd role manifest. `not_run`: detector/tracker inference,
forecast fit or scoring, risk calibration and independent
confirmation. No new CREATE job, no manuscript change, and no
change to the original 2% rule, deployment, Stage5C or SMC.
Raw annotations and video were not written to Git.

[Range index](range_index.json) |
[Support and overlap counts](analysis.json) |
[Real video smoke](video_prefix_probe.json)
