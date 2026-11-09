# Learning Signed Risk Recovers Coverage, Not Calibration

I retrained the two parts of the pooled decoder that the previous
experiment identified as bottlenecks. The direct risk head predicts
the signed all/easy excesses. The matched conditional head learns
these excesses only in their respective known subsets, and charges
the causal edit radius when outcomes are predicted wholly missing.
A separate gain head adds conservative preferences between edits on
different agents in the same query. All proposals and the decoder
stay fixed, and nothing from the held source enters model fitting.

## Recovered opportunities and realized recording risk

Of 2,453 repeated query views with an actual eligible seed, the frozen,
direct, and conditional heads recognize one **on the same candidate**
in **72, 1,442, and 1,843** views. They predict some seed in 73, 2,587,
and 2,837 views, respectively. The conditional head has much higher
recall but also admits false opportunities. Recovering a candidate
somewhere in a query is not the same as choosing it or passing risk.

| Risk / gain estimate | Edits | Original risk passes /72 | Undefined easy /72 | Worst known signed easy change |
|---|---:|---:|---:|---:|
| Frozen / frozen | 244 | 5 | 66 | +0.014% |
| Frozen / swap | 265 | 6 | 66 | +0.005% |
| Direct / frozen | 60,801 | 4 | 15 | +55.97% |
| Direct / swap | 57,084 | 2 | 16 | +44.92% |
| Conditional / frozen | 85,920 | 10 | 13 | +43.57% |
| Conditional / swap | 73,477 | 9 | 14 | +28.92% |
| Calibration-selected fallback | 0 | 0 | 72 | Undefined edited support |

The conditional head is a real coverage improvement, but **no
calibrator qualifies (0/9)**. The strongest ladder setting in any
cell clears only five calibration recordings. With frozen gain,
the conditional head gives +1.498 pp [0.159, 3.141] on known steps
versus CV, but the full-grid lower is -1.005 pp [-2.829, +0.315].
Against OLS4 it remains -54.910 pp [-96.411, -23.036] in conservative
CV-normalized utility. At the exact counts of the conditional pool,
the gain-only control has 0/72 risk passes versus 10/72; the pool's
direct lower contrast versus that control is -19.021 pp
[-40.927, -3.151]. The upper is +15.275 pp [+4.025, +30.864].
These are missing-label bounds, not a measured loss on the unavailable
future.

## Did the gain retrain help at the same coverage?

The unconstrained factorial swaps change intervention counts. I also
replayed both gain heads at the same achievable count per query
under the same risk head. Only the chosen recipients and paths can
change; amplitudes are not matched.

| Fixed risk head | Edits per arm | Frozen / swap risk passes | Swap known gain (pp) | Swap full-grid lower (pp) |
|---|---:|---:|---:|---:|
| Direct | 54,486 | 3 / 2 | +0.2378 [+0.1266, +0.3436] | -3.3957 [-7.7593, -0.4247] |
| Conditional | 71,595 | 8 / 9 | +0.2686 [+0.0195, +0.6067] | -1.3641 [-2.9569, -0.3814] |

Both improvements on known steps disappear under the paired
completion bound. The conditional upper contrast is +1.9014 pp
[+0.4600, +4.1259]. All intervals resample source groups 3,000
times after seed averaging. The swap comparison includes a new
gain fit, not a loss-only ablation. It is related to
[decision-focused learning to rank](https://proceedings.mlr.press/v162/mandi22a.html);
the pairs here are physical forecast alternatives on the same query,
and uncertain label directions receive no preference supervision.

## What remains to fix?

With conditional risk and frozen gain, **37** views already exceed
the easy positive-harm limit on known outcomes. Of the **22** that
pass all observed checks, **12** fail when wholly missing outcomes
are charged; 10 pass the original rule. Thirteen views have undefined
selected-easy support, counted separately. Thus this is both
underestimated known easy harm and a missing-outcome problem, not
just an overly strict probability cutoff. The next comparison should
separate a tail-aware loss for observed easy harm from a lead-specific
missing-step disagreement estimate in the utility head. Retain the
conditional head as a coverage control rather than as a deployment policy.

## Reproduction

`fresh_run`: 27 Torch fits, 27,000 updates, three seeds, 24 held
fitting recordings in nine source groups, 135.39 s on native arm64 CPU.
Four Torch threads, no loader subprocesses, checkpoint every 100
updates. `cached_verified`: frozen source-excluded forecasts, previous
244-edit control, exact zero-update model/choice replay. Forty-one
focused tests pass; the full legacy suite was not rerun.

```sh
.venv-pytorch/bin/python scripts/run_m3w_cv_signed_coverage.py
.venv-pytorch/bin/python scripts/run_m3w_cv_signed_coverage.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_signed_matched_gain.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_signed_failures.py
```

`not_run`: new A or DroneCrowd calibration outcome access,
independent confirmation, deployment, Stage5C, SMC. The original
2% rule and 15/30 split are unchanged. Native 8/12 annotation steps
and image pixels are not verified seconds or physical units.
This is not a true 3D or foundation model.

[Protocol](protocol.md) | [Factorial readout](readout.json) |
[Matched gains](matched_gain.json) | [Risk failures](risk_failures.json) |
[Manuscript v131](../evidence_manuscript_v131/README.md)
