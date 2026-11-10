# OpenDD: learning gain and harm did not improve matched allocation

I fit a new count head to continuous gains over CV and incremental gains over
OLS4, with a separate fit adding the three risk terms. The paths, nominees,
scorer and time budget were unchanged. The experiments used seven outer
location holds and three seeds. Each of the 42 small heads ran 500 updates;
donor nominations came from rankers that had seen neither the donor nor the
outer location. I ran this on native arm64 with four Torch threads and no
loader workers. The full fit completed in 21.9 seconds, then the checkpoint
replay reproduced the readout with zero further updates.

| Head | Online passes / 606 | Online edits | Count- and movement-matched passes / 606 | Matched original risk passes / 606 |
| --- | ---: | ---: | ---: | ---: |
| Frozen count head | 57 | 780 | 60 | 88 |
| Gain only | 54 | 1,533 | 48 | 81 |
| Gain and risk, penalty off | 56 | 1,562 | 57 | 89 |
| Gain and risk, penalty on | 60 | 1,292 | 42 | 73 |

A pass requires the original 2% risk checks and positive conservative gain
over **both** motion references. The 60 online passes are not an improvement
at fixed spending. The penalty-on arm's paired matched pass-rate difference
is -2.01 percentage points at equal location weight (95% location bootstrap
[-3.80, -0.52]). Its conservative gain versus the matched frozen plans is
-0.03075 pp (95% [-0.05851, -0.00254]). When offline allocations were unresolved,
they kept the original online plan (two for gain only, one for joint without
penalty, two with penalty); even assigning both penalty-on cases a pass would
leave it below the frozen 60.

Turning the penalty off recovers 15 matched passes with the same network.
This penalty did not turn the learned risk terms into safer decisions at a
fixed budget. The result does not establish that all risk-aware training is
ineffective: it tests these features, marginal targets and this penalty. The
comparison uses the whole frozen development roster, not independent external
evidence. I did not replace the policy, tune thresholds, or open confirmation
outcomes. The remaining question is whether learning which choices are safe
*relative to other available choices* can recover any of the known opportunity.

## Reproduce

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/run_m3w_opendd_prefix_value.py --replay
```

Omit `--replay` to fit or resume; the runner saves atomically every 100
updates. The [protocol](protocol.md), [registration](registration.json),
[checkpoint manifest](head_freeze.json) and [readout](readout.json) document
each held-location/seed cell. For the public scalar replay without local
assets, use the [v4 supplement](../allocation_review_supplement_v1/README.md).
