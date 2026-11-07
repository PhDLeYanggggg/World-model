# Learning A Local Forecast Around OLS4

I tested a bounded local forecast adjustment instead of adding another
switching threshold. The control sees the same agent history and uses
the same architecture and training draws, but its neighbor features
are zeroed. This is an OLS4-anchored proposal ablation, not an equivariant
interaction architecture like [EqMotion](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_CVPR_2023_paper.html).
The strong motion controls follow the evaluation lesson of
[Scholler et al.](https://arxiv.org/abs/1903.07933): beating a weak learned
forecast does not substitute for testing against a simple motion model.

## What Changed

I ran 84 fresh PyTorch fits: three seeds, 14 held source groups and
two matched variants. Each fold uses seven fitting sources and six
separate donor sources to choose how far to move from OLS4. The fit took
43.2 seconds on native arm64 CPU with two Torch threads and no loader
workers. We score eight observed and twelve forecast annotation steps
in image pixels on 34 already exposed recordings. The 102 recording-seed
views are not independent. The interval pairs the methods and resamples
the 14 source groups 3,000 times after averaging across seeds.

| Comparison | Conservative ADE gain (pp on CV denominator) | 95% source-bootstrap interval |
| --- | ---: | ---: |
| Donor-selected self vs. OLS4 | +7.639 | [+0.896, +14.616] |
| Donor-selected neighbor vs. OLS4 | +7.980 | [+1.082, +15.144] |
| Donor-selected neighbor vs. matched self | -3.764 | [-8.478, -0.388] |
| Fixed full-amplitude self vs. OLS4 | -2.806 | [-32.621, +18.777] |
| Donor-selected self vs. fixed common translation | -3.473 | [-8.579, +1.846] |
| Donor-selected self vs. CV | -2.337 | [-16.024, +7.925] |

Each comparison charges its own path disagreement on wholly unobserved
rows, so the conservative gains are not additive. A negative paired
lower bound does not establish a negative full-label effect. Against
matched self, the neighbor arm's conservative advantage is unsupported,
and the lower-bound contrast favors self. None of the full-path arms
passes any of the 102 original recording risk views. Small amplitude is
useful for the OLS4 comparison, but neither positive OLS4 contrast is
evidence of safe deployment or an improvement over CV.

## Does It Help As An Option?

Adding the frozen self proposal to the [previous twelve options](../native_slate_geometry_capacity_v1/README.md)
at exactly the same recipients leaves the capacity counts unchanged:
71 feasible views, 33 positive over OLS4, 31 infeasible, no unresolved
solves. The hindsight solver uses the proposal in 66 feasible views
but this does not create new passing views. The fixed common translation
remains a necessary matched comparator. No policy is changed, no
external calibration or confirmation is used, and Stage5C and SMC
remain off.

[Registered training](protocol.md) | [Accuracy and risk](analysis.json) |
[Follow-up protocol](capacity_protocol.md) | [Follow-up result](capacity_analysis.json)

```sh
.venv-pytorch/bin/python scripts/run_m3w_local_ols_proposal.py fit
.venv-pytorch/bin/python scripts/run_m3w_local_ols_proposal.py score
.venv-pytorch/bin/python scripts/run_m3w_local_ols_proposal.py replay
.venv-pytorch/bin/python scripts/audit_m3w_local_ols_capacity.py score
```

Fit resumes from per-cell checkpoints. Score and replay load the same
hash-bound frozen proposals. The capacity rerun verifies the private
per-record solver cache and light aggregate. Forecasts, solver witnesses
and checkpoints are not committed.
