# Okutama-Action: Annotation Cadence and Frozen Transfer

I tested whether the OpenDD joint score transfers without changing its
weights. [Okutama-Action](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w34/html/Barekatain_Okutama-Action_An_Aerial_CVPR_2017_paper.html)
contains scripted concurrent actions filmed by drones over one baseball
field. It is distinct from OpenDD, but single-site. The
[publisher's filename guide](https://github.com/miquelmarti/Okutama-Action/blob/master/index.md)
identifies simultaneous views by time of day and scenario. I grouped the
33 publisher-training clips into 17 such groups, not 17 independent sites
or participant groups.

## Source and Forecast Support

I fetched 11,417,124 bytes in five bounded requests from the publisher's
5.77 GB training ZIP. Its index lists 60,039 JPEGs; no image data were
fetched. All 33 tracking label files pass ZIP CRC and local SHA-256 checks.
The release has 333,720 tracking rows; 300,928 (90.2%) are flagged as
generated, and 18,817 as occluded. Of 30,911 visible nongenerated rows,
30,785 occur at frame numbers divisible by ten. This is a property of the
released tracking files, not a verification of their frame-to-time map.

Windows start on a fixed archive frame clock and do not overlap. An
agent enters from eight visible past positions alone. I exclude
lost/occluded boxes; the nongenerated analysis also excludes generated
boxes in history and future. I score partial futures and retain absent
futures in the roster. Results that use generated boxes are an
annotation sensitivity, not evidence from causal online inputs.

| Labels | Frame stride | Eight-step histories | Complete futures | Windows with two complete futures | Fully labeled rosters |
|---|---:|---:|---:|---:|---:|
| Released visible | 1 | 15,620 | 15,292 | 2,370 | 2,477 |
| Released visible | 4 | 3,634 | 3,241 | 531 | 448 |
| Nongenerated visible | 1 or 4 | 0 | 0 | 0 | 0 |
| Nongenerated visible | 10 | 1,175 | 499 | 91 | 62 |

The last row has 658 partial and 18 absent futures, and multi-agent
support in 16 scenario groups. It is an annotation-cadence diagnostic,
not a change to the primary benchmark. On released stride-one labels,
OLS4 beats CV by +18.10 conservative pp [14.78, 21.31]. On nongenerated
stride-ten labels it loses -26.12 pp [-46.29, -9.51], and on released
stride-ten labels it also loses (-24.54 pp). The reversal cannot be
attributed to the generated flag alone: changing the interval changes
the task. The past-prefix selector scores -14.02 pp [-26.88, -3.49]
on nongenerated keyframes. Displacements are 4K annotation pixels.

## Frozen Transfer on Nongenerated Keyframes

For each existing seed, I averaged the seven frozen OpenDD cost
heads and the seven relative-motion predictions. Okutama data do not
enter fitting or threshold selection. The same past-only rule
nominates at most two OLS4 peak slots per query. All policies fill
the same 774 slots across seeds using OLS4 or tapered turn at those
radii. The risk readout uses the original three 2% checks, with
easy/hard reference cuts from other publisher-training scenario groups
for scoring only.

| Policy | Conservative CV gain, pp [95% group interval] | Risk passes / edited views |
|---|---:|---:|
| Past-only OLS4 nominations | -3.71 [-7.14, -0.49] | 0/99 |
| Frozen additive | -3.24 [-6.50, -0.23] | 1/99 |
| Frozen risk-only joint | -3.28 [-6.53, -0.28] | 1/99 |
| Frozen relative joint | -3.15 [-6.42, -0.13] | 1/99 |

The relative head improves co-observed pair error over additive by
**+1.50 px [0.20, 3.35]**. After charging for missing pair steps, the
full-grid lower gain is **-3.51 px [-11.94, +1.53]**, with zero strict
positive pair wins. Intervals use 3,000 paired scenario draws after
averaging seeds within clip and cameras within scenario. They describe
variation within this facility, not transfer to new sites.

This is a separate label-history stress test, not confirmation of the
method. There was no online tracking, stabilization or metric
verification; camera motion remains in the image-plane paths.
No test payload, reserved outcome, policy promotion, Stage5C or SMC.

## Replay

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_okutama_label_histories.py acquire
.venv-pytorch/bin/python scripts/run_m3w_okutama_label_histories.py freeze
.venv-pytorch/bin/python scripts/run_m3w_okutama_label_histories.py score
.venv-pytorch/bin/python scripts/run_m3w_okutama_frozen_transfer.py freeze
.venv-pytorch/bin/python scripts/run_m3w_okutama_frozen_transfer.py score
```

After acquisition the replay is offline. The arm64 transfer uses
four compute threads and no workers. The PID heartbeat and each
frozen output are atomic. Rerun the interrupted phase to verify or
resume. Raw data, forecasts and weights are not distributed.
The source is [CC BY-NC-SA 3.0](https://github.com/miquelmarti/Okutama-Action/blob/master/index.md#license).

[Label and baseline results](readout.json) |
[Transfer registration](transfer_registration.json) |
[Transfer results](transfer_readout.json)
