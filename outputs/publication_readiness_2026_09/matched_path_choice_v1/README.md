# Can an OpenDD Head Choose the Matched Paths on TUMDOT?

The earlier TUMDOT study found a positive hindsight policy at fixed
recipients and radii. I tested whether an OpenDD-trained causal head
can realize that benefit without fitting or selecting on TUMDOT.

The same six path families are used throughout. On OpenDD, each donor
path comes from a producer and refiner that excluded the donor location.
The selector pools these donor predictions; they are selector fitting
data, not an independently held OpenDD selector evaluation. On TUMDOT,
each seed uses the fixed seven-model ensemble per candidate, whereas
OpenDD training uses one site-excluded prediction per candidate. This
train-to-inference discrepancy and the source shift are not separated.
There are 10,025 edited fitting rows per seed across seven locations.
Two locations contribute 9,447 rows; batches sample locations uniformly
before sampling rows. There are six native Torch fits, 3,000 updates,
three seeds, and no TUMDOT model selection or calibration. All heads
were frozen before scoring the nine already exposed recordings.

The head learns full-grid conservative gain and observed easy harm
per unit of matched radius, including pairwise cost contrasts. It sees
eight past steps and predicted path directions for twelve future steps,
not future positions or validity. The action-only ablation removes
explicit history and the CV endpoint but retains heading and speed
normalization of the candidate path. The primary rule uses a candidate
only when utility and easy-penalized rankings agree and strictly improve
predicted penalized utility over the causal hindcast. Otherwise it emits
hindcast at the same radius. This agreement rule is not a risk certificate.

| Half-radius policy | Conservative vs CV (pp) | Direct vs matched hindcast (pp) | Original risk passes |
|---|---:|---:|---:|
| OpenDD-best fixed (tangent in all three seeds) | -0.0533 | -0.1929 | 21/27 |
| Matched causal hindcast | -0.0365 | 0.0000 | 24/27 |
| Action-only, agreement | -0.0228 | -0.1052 | 22/27 |
| Full history, agreement | -0.0618 | -0.1661 | 21/27 |

Directly, full history loses **0.0712 pp** against action-only agreement
and **0.0399 pp** against donor-best fixed at half radius. Direct lower
bounds use the difference between the two paths for missing steps; they
are not differences of separately bounded scores. Full agreement loses
to matched hindcast on each of the three campaign days. Its six risk
failures are the three seeds on each of P3-d1 and P3-d2. Easy reference
support is defined in all views. Worst observed net easy degradation is
0.904%, but the worst conservative easy-risk bound is 3.682%; the maximum
all-row risk bound is 0.565%. Thus the old rule fails even though the
observed net easy metric alone looks acceptable. At quarter radius all
policies pass 27/27 but no policy beats CV or matched hindcast. There is
one campaign, not 27 independent groups; no independent-domain CI is
claimed. This refutes the tested head, not learnability with all causal
information. The prior +0.4245 pp hindsight witness remains diagnostic.

The result does not justify deployment, changing the 2% threshold,
or opening DroneCrowd confirmation. To improve on it, isolate the
single-producer/ensemble gap on development data before investing in
another flexible head, and compare with causal past backtests.

```sh
.venv-pytorch/bin/python scripts/run_m3w_matched_path_choice.py
.venv-pytorch/bin/python scripts/run_m3w_matched_path_choice.py --replay
```

[Registration](registration.json) | [Frozen heads](head_freeze.json) |
[All variants and day breakdown](analysis.json).
