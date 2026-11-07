# Correcting Donor Proposals Does Not Transfer Their Risk Advantage

The first [demonstration experiment](../readout.md) could not test risk
teaching: maximizing known gain gave the same choices with or without the
risk constraints. In this follow-up the teacher instead ranks current
actions by a causal utility proposal fitted without **both** the donor
and the held evaluation source. Only the constrained teacher filters
those proposals using the donor's observed risk. Its selected count
is given to both teachers. No donor outcome enters a student feature.

All 42 held-source/forecast-seed cells completed. This is 546 ordered
held/donor proposal fits and 84 student fits on the three frozen
forecaster banks. The constraint changes 32,297 of 61,849 active donor
query views (52.22%). Those are repeated donor views over different
exclusions, not independent samples. Each student's squared ranking
loss supervises membership in the chosen set; it does not fit the
ordering of complete sets directly.

| Evaluated Policy | Matched Switches | Completion Lower vs CV (pp) | Easy Risk Upper Above 2% | Undefined Easy | Original Passes / 102 |
|---|---:|---:|---:|---:|---:|
| Original causal ledger | 6,184 | +0.10154 | 65 | 34 | 3 |
| Student of utility-only proposals | 6,184 | +0.11261 | 64 | 36 | 2 |
| Student of risk-filtered proposals | 6,184 | +0.10822 | 64 | 36 | 2 |

**The matched primary does not improve.** The risk-filtered student
replaces 1,375 of the utility-only student's selected occurrences.
Its direct paired conservative difference is -0.00528 pp
[-0.01425, +0.00240], and its known difference is -0.00256 pp
[-0.01064, +0.00393]. Against the original, the direct lower is
-0.00081 pp [-0.01410, +0.01467]. Subtracting separate lower-vs-CV
values does not produce the paired lower: the latter penalizes
between-policy disagreement on wholly unknown outcomes. Intervals use
3,000 paired draws of 14 sources after averaging actual bank seeds.

Both students have 70 all-risk violations and three signed easy-net
violations. Hard known gains against CV are +0.05187% for risk-filtered,
+0.05635% for utility-only, and +0.05464% for the original. The local
relative-position error reduction is +0.00063 image-coordinate units
per supported query [-0.00183, +0.00311] against the utility-only student.
There is no established interaction or safety benefit.

The experiment distinguishes two failures. An oracle that maximizes
realized utility is a degenerate teaching control; a fallible proposal
creates informative differences, but those differences do not confer
observed risk control on the fitted membership ranker. This does not
identify whether representation, marginal supervision, or transport
of the risk estimates is the limiting factor. A cleaner next
comparison scores the actual rejected and repaired **sets**, holding
their counts and candidates fixed.

[Mandi et al., ICML 2022](https://proceedings.mlr.press/v162/mandi22a.html)
rank feasible solutions in decision-focused learning; our current
student instead distills action membership from an outcome-constrained
teacher. [Elmachtoub and Grigas, Management Science](https://doi.org/10.1287/mnsc.2020.3922)
optimize a downstream decision loss. Neither result gives a guarantee
for our recording-relative risk ratios. These comparisons are method
positioning, not claimed reproductions or a novelty claim.

The population remains exposed development data, eight observed and
twelve future annotation steps in image coordinates. Original support
and 2% rules are unchanged. No independent outcomes, physical safety
claim, deployment, Stage5C or SMC.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_nested_feasible_teacher.py fit
.venv-pytorch/bin/python scripts/run_m3w_nested_feasible_teacher.py score
.venv-pytorch/bin/python scripts/run_m3w_nested_feasible_teacher.py replay
```

Protocol `920cf6b9` was committed before fitting. The native arm64 matrix
takes 75.72 s and checkpoints each held source. The original control and
complete scoring replay exactly. Forty-seven scoped tests pass.
[Protocol](protocol.md) | [Replay receipt](replay_execution.json)
