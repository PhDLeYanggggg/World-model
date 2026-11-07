# Native Transport Candidate Rehearsal

**Decision:** Do not admit the new controller to external confirmation.
The 2% risk limit, reference support and handling of undefined risk are
unchanged. No identity exception was introduced. The 112-recording
DroneCrowd reservation is unchanged; neither its inputs nor its outcomes
were opened by this experiment.

The new decision rule screens joint switches using predicted gain, predicted
all/easy risk excess and a predicted positive easy reference, all for the
current query. The [registered protocol](protocol.md) fixes the candidate
bank, the training-only cuts (12.492 and 61.718 pixels), and the separation
between the CV risk reference and OLS4 accuracy comparator.

Fresh controller fitting used three seeds. There are 42 leave-one-source-out
fits for rehearsal and three pooled fits frozen for provenance only. The
forecast bank is reused with verified lineage. All 34 recordings from
14 development sources were previously exposed. The 102 recording-seed
views are not 102 independent recordings. Intervals use 3,000 paired
source-group draws after seed averaging.

| Policy | Switches | Original rule passes / 102 | Undefined easy risk | All-risk upper bound >2% |
| --- | ---: | ---: | ---: | ---: |
| CV | 0 | 0 | 102 | 0 |
| OLS4 | 375,510 | 0 | 0 | 102 |
| Independent | 9,675 | 4 | 23 | 88 |
| Scene family | 9,675 | 1 | 24 | 88 |
| Pair additive | 9,675 | 3 | 22 | 90 |
| Joint | 9,675 | 3 | 24 | 87 |
| Risk-screened joint | 90 | 3 | 84 | 15 |

The screened policy's paired completion-lower utility versus CV is
**+0.00524 pp** [0.00006, 0.01455]. Against OLS4 it is **-9.819 pp**
[-21.997, -0.550] with observed-label utility +3.964 pp [-0.680, +8.631].
The conservative lower bound does not establish a win over OLS4.
Risk screening removes over 99% of interventions, but leaves nearly all
views without a pass. The apparent safety gain from switching less is not
a deployment result; unsupported views still fail to establish safety.

At matched counts, joint over additive gives -0.0162 pp
[-0.0346, -0.0001] on observed-label absolute utility. Observed relative
pair ADE reduction is +0.013 pixels [-0.009, +0.045]; the missing-step
bounds span -0.158 to +0.175 pixels. This does not establish a benefit
from the joint pair term. Improve the model's estimates of reference
support and harm before spending the independent evaluation.

## Reproduction

Run these commands with the existing verified local development assets:

```sh
.venv-pytorch/bin/python scripts/freeze_m3w_dronecrowd_transport_candidate.py freeze
.venv-pytorch/bin/python scripts/freeze_m3w_dronecrowd_transport_candidate.py score
.venv-pytorch/bin/python scripts/freeze_m3w_dronecrowd_transport_candidate.py replay
```

[Metrics](analysis.json), [freeze receipt](freeze_execution.json),
[score receipt](score_execution.json), [replay receipt](replay_execution.json).
The receipts bind aggregate results and ignored local coefficient files.
This is native observation-8 / prediction-12 in dataset pixels, not seconds,
metric, or 3D evidence. Stage5C was not executed and SMC remains off.
