# P-DESTRE: box format was the first blocker, not the last

The [source audit](../pdestre_source_intake_v1/README.md) showed
that P-DESTRE supplies usable video and many complete reference
histories. I tested whether those histories can be recovered from
the camera input itself. The first comparison exposed a mismatch
between the published box schema and the archived numbers.

## Frozen camera comparison

I use one native-frame window per acquisition-date prefix, from the
first lexically ordered eligible clip and its first query with two
complete observed histories. Future label support does not select a
clip, track or window. All 11 registered windows and 88 observations
completed. The dates are repetitions within **one linked source**,
not 11 independent sites or test groups. No published boxes or IDs
were passed to the detectors or tracker.

Both frozen detectors see identical decoded images. RAPiD uses its
existing whole-image input and 0.30 threshold. The COCO MobileNet
detector uses its existing 4-by-3 tiling and 0.35 threshold. Both
use the same online box association and eight consecutive observations.
An online identity must match the same known reference ID by
one-to-one IoU >=0.50 at every observed frame before it can be scored.
Unknown reference identities cannot certify a match; unmatched online
histories remain in the denominator. The detector and resolution
choices differ together, so this is not an architecture-only ablation.

## Format diagnosis

The [P-DESTRE release page](https://p-destre.di.ubi.pt/download.html)
and [paper](https://www.di.ubi.pt/~hugomcp/doc/PDESTRE.pdf) describe
the fields as x, y, height, width. That is the registered first
interpretation. It produces just ten RAPiD matches and one MobileNet
match among 758 known-ID past boxes, and makes 736 of those boxes
wider than tall. I kept that result and replayed the exact same
camera outputs, labels, frames and IoU rule with the size fields
reversed. No model, detector threshold or track was changed.

| Size interpretation | Detector | Past box matches / 758 | Online histories | Stable matches | Complete futures / 85 reference tracks | Whole roster complete |
|---|---|---:|---:|---:|---:|---:|
| Documented height, width | RAPiD | 10 | 108 | 0 | 0 | 0/11 |
| Documented height, width | MobileNet | 1 | 184 | 0 | 0 | 0/11 |
| Width, height diagnostic | RAPiD | 651 | 108 | 70 | 70 | 0/11 |
| Width, height diagnostic | MobileNet | 697 | 184 | 62 | 61 | 0/11 |

This is strong evidence of a size-field discrepancy, not a revised
official benchmark or a result selected for deployment. An independent
geometry check or publisher confirmation should precede any formal use.
There are 88 complete reference histories and 85 with all 12 future
labels. In the alternate layout, RAPiD recovers complete pairs in 10
windows and MobileNet in 8. More framewise matches do not imply more
stable tracks: on the same framewise assignments, ignoring online
identity would leave 72 and 73 complete references respectively.
RAPiD still emits 38 histories without a scored future; MobileNet
emits 122. These are unsupported forecasts, not proven false positives.

## Motion controls, complete matched targets only

| Frozen detector | Matched targets | CV ADE / FDE (px) | OLS4 ADE / FDE (px) |
|---|---:|---:|---:|
| RAPiD | 70 | 35.57 / 61.95 | 25.02 / 40.73 |
| MobileNet | 61 | 50.52 / 84.81 | 34.26 / 52.38 |

Each CV/OLS4 row is paired on identical targets. The two detectors
score different target subsets and cannot be ranked by these errors.
OLS4 has lower FDE in 7 and 10 of the 11 windows respectively.
Coordinates are box-centre pixels and the horizons are native frame
indices. This does not establish whole-scene utility, risk, camera
calibration or seconds-level results for another dataset.

**Decision:** a format fix recovers many pairs, but neither pipeline
has a complete online roster. Keep P-DESTRE as a camera/association
stress test. Do not use these 11 windows as an independent or
whole-roster comparison of joint versus independent forecasting.
Prioritize a genuinely separate source with a verified geometry
contract and complete reference support for the main comparison.

## Replay

```bash
export SSL_CERT_FILE=/etc/ssl/cert.pem OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_pdestre_camera.py freeze
.venv-pytorch/bin/python scripts/run_m3w_pdestre_camera.py score
.venv-pytorch/bin/python scripts/audit_m3w_pdestre_box_order.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_pdestre*.py tests/test_m3w_external_causal_sources.py tests/test_m3w_kit_ais.py tests/test_m3w_topdown_source_support.py
```

The native arm64 CPU run used four Torch threads, zero loader
workers, and per-window atomic receipts. Freeze took 321.30 s
at 3.74 GB peak RSS; scoring took 5.90 s. The freeze was pushed
before the first geometry read. Resume by rerunning the same
phase after checking the original process has exited.
No images, video, raw labels or per-person prediction arrays
are committed. 22 focused tests passed.

[Registration](registration.json) | [Freeze](freeze.json) |
[Documented-order readout](analysis.json) |
[Size-order diagnostic](box_order_diagnostic.json).
No new training, CREATE job, independent confirmation, policy risk
scoring, deployment, manuscript edit, Stage5C or SMC.
