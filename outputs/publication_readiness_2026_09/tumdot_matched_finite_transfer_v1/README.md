# Frozen Forecast Transport at Matched Radii

**Question.** Does the OpenDD CV-tangent direction or finite-harm training
transfer to TUMDOT-ING at the same intervention recipients and per-row
peak displacements as the incumbent and a causal past-motion hindcast?

**Scope.** Fresh native Torch inference from hash-verified frozen checkpoints;
no new training, target-domain calibration or model selection. Nine previously
exposed recordings, three training seeds, 404 query windows, 1,456 target rows,
1,259 complete futures and 17 wholly missing futures. The recordings span
three nearby intersections over three days of one campaign, not 27 independent
sites. The original 8-observation/12-prediction transfer scores replay exactly
in all 27 recording-seed views. No access to DroneCrowd confirmation data.

| Fixed policy | Half: lower utility vs CV (pp) | Half: original risk passes | Quarter: lower utility vs CV (pp) | Quarter: original risk passes |
|---|---:|---:|---:|---:|
| Incumbent | -0.1160 | 24/27 | -0.0567 | 27/27 |
| Earlier shape | -0.1066 | 24/27 | -0.0516 | 27/27 |
| CV tangent | -0.0533 | 21/27 | -0.0228 | 27/27 |
| Finite harm | -0.0578 | 24/27 | -0.0267 | 27/27 |
| Curvature guard | -0.0581 | 24/27 | -0.0269 | 27/27 |
| Past-motion hindcast | -0.0365 | 24/27 | -0.0169 | 27/27 |

Radii are fractions of the realized peak of the original frozen 5% blend,
not alternative mixing coefficients. At half radius, direct conservative
comparisons give tangent +0.0173 pp against the incumbent, +0.0101 against
earlier shape and -0.1929 against matched hindcast. Finite harm loses
0.0573 pp to tangent directly. A direct bound debits missing steps using
the paired prediction difference and cannot be reconstructed by subtracting
two bounds against CV. Against CV, tangent gains +0.1032 pp on observed
labels but incurs 0.1357 pp on partially missing rows and 0.0208 pp on wholly
missing rows under the original conservative bound. Against hindcast it also
loses slightly on observed labels (-0.0062 pp with the same CV denominator).

**What is still possible?** All views have easy reference support. At half
radius, the minimum-observed-ADE hindsight selector attains both the 2% risk
pass and positive conservative CV utility in 27/27 views with either the
three-path menu (incumbent, hindcast, earlier shape; +0.3260 pp mean lower
utility) or the expanded six-path menu (+0.4245 pp). The exact maximum CV
lower-score selector attains +0.4338 pp with 25/27 risk passes for the expanded
menu. These are separate optimization objectives, not a joint optimum. The
minimum-risk result is a joint witness, not a forecast or a deployable policy;
it uses target labels to choose a path. The unchanged fixed arms cannot find
that witness causally. Shrinking the radius alone meets risk without useful
transfer. Do not tune a policy on these exposed recordings and report it as
new confirmation.

```sh
.venv-pytorch/bin/python scripts/run_m3w_tumdot_matched_finite_transfer.py
.venv-pytorch/bin/python scripts/audit_m3w_tumdot_matched_transfer_capacity.py
```

[Frozen-arm results](analysis.json) | [Hindsight and debit breakdown](capacity.json) |
[Source and checkpoint bindings](registration.json).
This is a development transport result, not a deployment, metric/seconds,
true-3D or foundation-model claim. Stage5C and SMC remain off.
