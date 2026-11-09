# OpenDD: Better Local Rankings Do Not Give Better Joint Forecasts

Whole-pair admission failed on the previous OpenDD run because the
model rarely ranked a feasible pair first. I kept the same paths,
two-slot budgets, source roles, easy references and calibration rule,
and changed how the selector learns to order the candidates.

## Controlled Refit

I trained 42 CPU models: two variants for each of seven held
locations and three seeds. Within each fold, the adjacent
calibration location and the evaluation location are excluded
from the five fitting sources. Both variants keep the original
natural-sample loss and see the same extra donor-only examples:
up to four positive slates per query with their two closest
negative slates at the same radii. Mining favors negatives with
easy reference support, so that recognizing missing support
alone does not solve the training task.

The mined BCE control uses those examples pointwise. The ranking
variant additionally penalizes a positive slate receiving a lower
score than its same-query negative. This is a RankNet-style logistic
ordering loss ([Burges et al.](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf)),
not a novelty claim. The specific question is whether supervision
within exact matched motion slots improves the selection of
risk-feasible forecasts without sacrificing relative motion.

## Results

| Method | Feasible pair ranked first: calibration | Feasible pair ranked first: held | Forced joint risk passes / 159 edited views | Forced joint minus additive full-grid pair gain |
|---|---:|---:|---:|---:|
| Original pointwise BCE | 15/174 | 11/180 | 9 | −0.001910 |
| Mined BCE | 24/174 | 19/180 | 17 | −0.002598 |
| Mined BCE + pairwise ranking | 27/174 | 27/180 | 16 | −0.006109 |

The denominators are query-seed views with at least one locally
feasible option, not independent scenes. Ranking over mined BCE
raises held top-one capture by 2.97 pp under equal-location
weighting; the 3,000-draw location bootstrap interval is
[0.00, 4.90] pp over three supported locations. The calibration
estimate is +1.42 pp [−1.08, 5.33]. This is a modest learning signal,
not a stable cross-source win.

Both new policies are forced to use the same 966 intervention
slots as the original study. Under ranking, the full-grid pair
interval is **[−0.010241, −0.001977]** in native coordinates. The
comparable mined BCE interval is [−0.004057, −0.000999]. The
ranked joint policy's conservative CV gain is **−0.1052 pp**
[−0.2195, −0.0126]; known ADE/FDE are 0.54786 / 1.25465,
versus CV at 0.54726 / 1.25329. Its mean recording p95 ADE
is 1.38949 versus CV at 1.38783. The hard point estimate is
−0.056% and 49 edited views exceed the signed easy 2% limit;
52 have an undefined edited-easy reference. Ranking yields
five strict positive pair views, but those do not offset the
negative average.

Neither model produced an admissible nonempty release in any of its
21 calibration folds. Each was also checked at all 483 nonempty score
breakpoints on the calibration locations, with zero passing. Both
calibrated arms returned CV. The matched interaction comparison
after fallback is not estimable, not a passed safety check.

## Target Check

The ranking target asks for local accuracy and risk feasibility;
it does not prefer a feasible slate that improves pair-relative
error over the matched additive slate. I audited that distinction
on the calibration menus only. Of the **174** feasible views,
**144** contain a feasible slate that also improves the full-grid
pair score without harming conservative accuracy versus additive.
The ranking head finds one of those slates in only **15** views.
This is a hindsight local opportunity count, not evidence that a
whole recording would pass. But it shows why improving the local
classification target alone need not improve joint decisions.

Next, learn and compare the pair-relative objective *within* the
feasible set, against a fixed additive reference. Keep the
accuracy and risk requirements separate; do not replace them
with the pair metric or adjust thresholds on these held folds.
No deployment or independent-confirmation decision changes.

## Reproduce

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_pair_ranking.py freeze
.venv-pytorch/bin/python scripts/run_m3w_opendd_pair_ranking.py score
.venv-pytorch/bin/python scripts/audit_m3w_pair_ranking_targets.py
```

The arm64 fit, calibration and freeze took 56.22 seconds (under
763 MB peak RSS). Scoring took 17.22 seconds. Four Torch threads,
zero loader workers; optimizer, weights and sampler are saved
every 50 steps, with a phase/PID heartbeat. Twenty focused tests
passed. No CREATE job, Stage5C, SMC or reserved outcome access.
[Registration](registration.json), [metrics](summary.json),
[target audit](target_audit.json).
