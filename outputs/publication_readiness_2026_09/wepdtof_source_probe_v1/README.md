# WEPDTOF: can real camera histories support an external forecast test?

I wanted a new camera source with more complete future labels than
Moving, without feeding edited annotation tracks to the model. I used
the [WEPDTOF publisher archive](https://vip.bu.edu/projects/vsns/cossy/datasets/wepdtof/download/)
and grouped the 16 clips by the 14 original videos described in the
[WACV paper](https://openaccess.thecvf.com/content/WACV2022/html/Tezcan_WEPDTOF_A_Dataset_and_Benchmark_Algorithms_for_In-the-Wild_People_Detection_WACV_2022_paper.html).
The two Large Office clips stay together, as do the two Jewelry Store
clips. I fetched the annotation members and a small set of pilot images
by verified byte ranges instead of downloading the 5.44 GB archive.

## Label support

The 10,544 released images contain 93,364 annotation rows. I found
558 ambiguous frame/identity pairs (557 Call Center, one Printing
Store) and excluded both rows at each pair rather than choosing an
identity. IT Office has 81 released images with no annotation-table
entry; those labels remain unavailable. The two initial incomplete
audits and their registration receipts are preserved. The final audit
uses identities and frame presence, not coordinates, for every clip.

| Released-frame stride | Nonoverlapping queries | Past-eligible targets | Complete 12-step futures | Queries with at least two complete targets |
| --- | ---: | ---: | ---: | ---: |
| 1 | 497 | 4,486 | 4,453 | 497 |
| 4 | 123 | 1,093 | 1,065 | 123 |
| 8 | 58 | 498 | 479 | 57 |

At stride four, 13 sources support a multi-agent query. Eligibility
uses eight observed labels; future availability is counted separately.
This is not a measurement of online detector coverage or a claim that
the overlapping population has 1,093 independent people. The release
grid is not converted to seconds and the coordinates are image pixels.

## Camera pilot

I fixed 12 disjoint windows on Convenience Store using frame names
alone. In each window, both detectors see exactly eight images, one
at a time. The tracker and CV/OLS4 forecasts use those images only.
The ordinary COCO MobileNet and [frame-only RAPiD](https://github.com/duanzhiihao/RAPiD)
use the same tracker, target matching and score. The published RAPiD
checkpoint lists COCO, MW-R and HABBOF training, not WEPDTOF; the
pilot does not certify image-level pretraining disjointness.

| Same 96 images | Detection rows | Online 8-step histories | Stable matches / 72 reference histories | Matched windows |
| --- | ---: | ---: | ---: | ---: |
| MobileNet | 363 | 27 | 22 | 11/12 |
| RAPiD | 522 | 58 | 57 | 12/12 |

Both detectors have complete futures for all stable matches.
The coverage increase is not automatically a forecasting gain:
the 57 and 22 matched histories are different populations. I
therefore repeated the comparison on the **same 20 targets**
in 10 windows that both detectors matched:

| Source of the past history | CV ADE / FDE (px) | OLS4 ADE / FDE (px) |
| --- | ---: | ---: |
| MobileNet camera | 15.22 / 25.69 | 10.59 / 16.48 |
| RAPiD camera | 14.99 / 25.01 | 11.05 / 17.29 |
| Publisher annotation diagnostic | 5.88 / 12.40 | 6.93 / 13.79 |

**The baseline ranking reverses** when I replace camera histories
with annotation histories on those same targets. This is why I
would not pick the motion baseline using annotated trajectories
alone. It is one development camera, not a population effect or
a causal estimate of detector error. The annotation diagnostic is
not an online result. No learned forecasting policy or risk pass
was scored.

## Frozen next step

The approved [roles](external_roles.json) keep Convenience Store
for development and reserve the other 13 original sources for one
frozen evaluation; 12 have structural stride-four support.
The [blind camera plan](blind_camera_plan.json) fixes 115 disjoint
windows and 920 past frames across the 15 external clips using
the publisher image catalog alone. It does not contain label
positions or predictions. No external camera inference has run.
I have not scored those sources or selected a model on them. The
original 2% rule and DroneCrowd 15/30 assignment remain unchanged.
See the [next comparison](next_experiment.md) for the prediction
and risk-rule freeze still required before opening outcomes.
No deployment change, Stage5C or SMC.

**Fresh run:** full label-support audit, 96 matched camera images,
two frame-only detector runs and one cold window per detector.
Both cold runs matched exactly. MobileNet production including
range retrieval took 254 seconds; RAPiD on cached images took 57
seconds, so these elapsed times are not a paired speed benchmark.
Both used native arm64 CPU, four threads and no data workers.
153 focused tests passed. The checkpoints and all source images
and annotations stay local.

[Support](support.json) | [Camera comparison](rapid_comparison.json) |
[Matched input contrast](input_contrast.json) |
[Chinese reproduction](../../../reproducibility/M3W_CURRENT_ZH.md) |
[Manuscript v158](../evidence_manuscript_v158/README.md)
