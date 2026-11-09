# AV2: Learning the Additive Control Does Not Recover Joint Utility

The [paired ceiling](../av2_supported_gain_v1/README.md) showed
57 two-slot views where two legal paths could satisfy the original
checks, but only two when the old additive path was fixed. I trained
a replacement additive control and compared it with a joint policy
using the same trajectories, recipients available to choose from,
two edit peaks, and risk and accuracy rules.

## Shared Inputs and Split

The 128 AV2 publisher-training scenarios use the same 8/12 grid in
native coordinates. Each outer fold has four donor cities, a separate
calibration city and a held development city, with seeds 17, 29
and 43. For joint training, the additive forecast of each donor
city comes from a model trained on the other three donor cities.
It has not seen that donor's labels, the calibration city or the
held city. There are 72 distinct action fits after deduplicating
identical donor sets and 18 pair fits. No weights are pretrained
on the held city. Both old fixed controls reproduce their earlier
readout exactly.

- **Point additive:** sum the two independently estimated action
  gains relative to CV and OLS4. Neither action scorer sees
  another agent's future or a pair reward.
- **Support additive:** subtract predicted positive all/easy risk
  excesses and signed easy excess per action. Prefer a slate
  containing a predicted easy recipient when one is available.
  If none is available, still fill the original slots in the
  forced comparison. The easy logit uses only recipient history
  features and the donor-derived cut, not the proposed action.
- **Joint:** re-rank up to eight support-ranked legal
  assignments with the conditional qualifying pair-gain head.
  The support additive forecast is the common reference. The
  comparator and the joint arm use the same two peak offsets.

Training supervises the two signed lower-gain increments and
original risk terms. It uses a local per-action approximation for
signed easy excess; that does not change the full-scene metric
used for calibration and evaluation. The confidence score for the
additive diagnostic is heuristic, not a risk certificate. Original
singletons are unchanged in the forced comparison. Two-arm release
uses only a threshold selected in calibration, and both arms must
meet the unchanged risk and dual-accuracy checks there.

## Forecast Results

| Policy | Known ADE | Available FDE | Conservative gain vs CV (pp) | Hard gain vs CV (%) | Risk passes / edited views | Undefined edited-easy |
|---|---:|---:|---:|---:|---:|---:|
| CV | 0.25617 | 0.57731 | 0 | 0 | No edits | No edits |
| OLS4 | 0.30431 | 0.65016 | -29.30 | -14.42 | Unmatched | Unmatched |
| Fixed additive | 0.26175 | 0.58323 | -1.88 | -0.40 | 5/285 | 252 |
| Fixed joint | 0.26084 | 0.58323 | -1.80 | -0.09 | 8/285 | 242 |
| Point additive | 0.26157 | 0.58326 | -2.12 | -0.07 | 5/285 | 246 |
| Support additive | 0.26220 | 0.58450 | -2.57 | -0.47 | 6/285 | 233 |
| Joint re-ranking | 0.26278 | 0.58780 | -2.84 | -0.69 | 8/285 | 228 |

All five matched policies make 459 edits. Compared with the fixed
additive path, the support model adds edited-easy support, but its
forced conservative gain and ADE are worse. Signed easy degradation
exceeds 2% in 25 fixed additive, 34 support and 41 joint views.
City-weighted scenario p95 ADE is 0.71878, 0.72102 and 0.72071
respectively. CV remains the strongest tested trajectory floor.

At the same peaks, the new joint policy's full-grid pair gain is
**-0.005257** against the learned support control, 95% interval
**[-0.019295, +0.005825]**. On the 59 fully labelled scenarios it is
-0.007505 [-0.021683, -0.000047]. Against the old frozen joint
forecast its all-scenario gain is -0.008812 [-0.015545, -0.004001].
These are direct paired contrasts, with 3,000 city bootstrap draws
after averaging seeds within scenarios. No lift is demonstrated.

## Why No Policy Clears Calibration

| Strict selection stage | Calibration views / 174 | Held views / 174 |
|---|---:|---:|
| At least one legal action set has edited-easy support | 114 | 114 |
| Point control has edited-easy support | 33 | 32 |
| Support control has edited-easy support | 41 | 45 |
| Support control passes both motion floors and all risk checks | 4 | 1 |
| A joint candidate meets the strict contrast against support | 55 | 53 |
| Strict candidate retained in the top eight | 11 | 10 |
| Strict candidate selected | 6 | 4 |
| Strict candidate selected and support control also passes | 0 | 0 |

The replay loads all checkpoints and reconstructs the held
predictions exactly. Only one held view has both a qualifying
support control and a qualifying joint alternative; the top eight
does not retain that alternative. Both the learned reference and
retrieval remain limiting. Under the original rule, **0/174**
nonempty thresholds pass. Even the auxiliary additive-only check
has no passing nonempty threshold (0/174). The released output is
identical to CV. There is no new risk certification or independent
confirmation.

I would not train another ranker on these same 128 scenarios first.
The next test should increase the number of development-only
scenarios from the publisher's train archive, with the selection
fixed before reading future labels and the same role separation.
Measure whether more support examples improve the additive control
before attaching the joint head. Keep a matched public 8/12
forecaster comparison and the untouched confirmation set as
separate requirements.

## Reproduce

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_learned_reference.py fit
.venv-pytorch/bin/python scripts/run_m3w_av2_learned_reference.py score
.venv-pytorch/bin/python scripts/audit_m3w_av2_learned_reference.py
```

The native arm64 Mac used four Torch threads and zero data workers.
Fresh training took 36.29 s and scoring 5.41 s, below 698 MB
RSS. Checkpoints every 40 updates store optimizer and sampler
state; the same command resumes. Phase and PID go to a heartbeat.
Twenty-nine focused tests passed. No CREATE job, AV2 official
validation/test, reserved confirmation outcomes, Stage5C, SMC,
deployment or manuscript changes. [Registration](registration.json),
[fit receipts](freeze.json), [metrics](summary.json),
[attribution](attribution.json).
