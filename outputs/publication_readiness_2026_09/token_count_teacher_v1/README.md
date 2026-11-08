# Training on the Exact Online Credit Rule

**Outcome: no evidence of a better count policy.** The preceding [capacity
audit](../streaming_menu_capacity_v1/README.md) finds 330/606 verified
positive hindsight cases even with the fixed nested menu and chronological
credit. This tests whether training on plans that obey chronological
credit recovers that gap. These are OpenDD development recordings,
not independent confirmation.

Both heads have the same nested action nominations, credit-state feature,
training architecture and causal online decoder. Pair-excluded donor
rankers and donor-only cutoffs are verified from the previous experiment.
The control projects an aggregate-quota plan onto the chronological rule
by dropping unaffordable actions. The other teacher optimizes with the
chronological rule inside its integer program. Its witnesses are replayed
and rescored under the unchanged CV-relative 2% risk rule. A donor
recording is omitted in both arms if either optimization is unresolved.
Inference only selects the logit at the current available credit state.

| Teacher | Edits | Risk pass | Risk pass and beat both | Undefined easy | Gain vs CV (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Projected aggregate plan | 944 | 69/606 | 46/606 | 380 | +0.111 |
| Exact chronological plan | 1,109 | 70/606 | 45/606 | 367 | +0.057 |

Direct exact-minus-projected conservative gain is **-0.070 pp**
(95% interval **[-0.182, +0.045]**, 3,000 location bootstrap draws after
averaging seeds). The credit rule is the same, not the realized counts.
The projection removes 3,826 edits across overlapping donor training
appearances. Exact labels verify 1,972 positive witnesses, versus 2,290
before projecting the aggregate labels. Neither the cleaner target
nor the shared causal credit feature closes the observed gap. The
experiment does not isolate the separate effect of including credit as
a feature relative to the older controller.

Imitating a single hindsight-optimal plan may discard useful information
about the relative value and risk of alternative edits. Test that
discrimination directly against a simple causal ranking before training
another plan imitator. The original risk rule, current deployment, and
DroneCrowd 15 calibration / 30 sealed confirmation groups are unchanged.
Stage5C was not executed and SMC was not enabled.

`fresh_run`: 42 native arm64 CPU Torch fits, 21,000 updates, three seeds,
four threads and zero workers. `cached_verified`: donor checkpoint reuse
and zero-update checkpoint replay. `not_run`: independent confirmation.
[Metrics](analysis.json) | [Registration](registration.json).

```sh
.venv-pytorch/bin/python scripts/run_m3w_token_count_teacher.py
.venv-pytorch/bin/python scripts/run_m3w_token_count_teacher.py --replay
```
