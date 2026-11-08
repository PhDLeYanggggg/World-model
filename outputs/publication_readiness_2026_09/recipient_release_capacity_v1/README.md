# Changing Recipients Recovers Capacity, Not a Causal Policy

**Fresh fitting-only diagnostic.** The previous convex bound excluded
all 24 recordings at fixed recipients and half-strength paths.
Here we hold the original 69,746 target rows, nine source groups,
future masks, CV fallback and 2% risk rule fixed. The exact-query arms
preserve all 31,923 genuine changes, but can choose different people.
The fixed-total arm preserves each recording's total, but can
redistribute actions across windows. Free count may intervene more
or less. None of these searches opens calibration or confirmation.

| Path bank and intervention budget | Verified both | Risk infeasible | Negative upper bound | Unresolved |
|---|---:|---:|---:|---:|
| Four half blends, per-query exact | 8 | 15 | 1 | 0 |
| Full strengths added, per-query exact | 8 | 15 | 1 | 0 |
| Two capped trend paths added, per-query exact | 8 | 15 | 1 | 0 |
| Expanded bank, recording total exact | 14 | 6 | 1 | 3 |
| Four half blends, free count | 12 | 5 | 7 | 0 |
| Half and full straight paths, free count | 18 | 5 | 1 | 0 |
| Expanded bank, free count | 18 | 5 | 1 | 0 |

Verified both means a constructive allocation passes the exact
original risk scorer and strictly improves the direct conservative
comparison to both CV and OLS4. The [SciPy MILP solver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)
is used with a maximin objective over those two gains; negative dual
bounds and reported infeasibility are distinct. A time-limited
unresolved search is not a proof of infeasibility. All positive
witnesses re-score under original masks and budgets; replay checks
witnesses and does not independently rerun solver proof searches.

## What makes the difference

- **Recipients matter without increasing the budget:** 0 becomes 8
  when people can change at the original per-query counts. No
  full-strength or curved path adds another pass at those counts.
- **Window allocation matters even at fixed recording totals:** the
  expanded bank adds six verified recordings, all from two fitting
  groups. The equal-source paired increase is +11.64 pp
  [0.00,+27.51], 3,000 bootstrap draws. Three fixed-total problems
  remain unresolved, so 14 is a verified count, not an exact maximum.
- **Amplitude helps at free count, not the curved alternatives:**
  12 becomes 18 with full strengths; curved paths do not add a pass.
  The 18 free-count witnesses change 47,994 paths, compared with
  31,354 original changes on those recordings. Sixteen use more
  changes and two use fewer. This is not a gain from universal
  sparsification.

The free-count bank still fails in six recordings. Three have no easy
reference anywhere in the roster, so no action can supply the missing
denominator under the unchanged rule. Two have easy rows, but no
candidate can change one within the 2% easy bound. The remaining
recording has a negative upper bound of -5.860 pp on the smaller
of the two conservative gains. None of these conditions is relabeled
as a pass.

## Real inference

The [leave-source-out cost selector](../recipient_release_crossfit_v2/README.md)
does not consume hindsight allocations. Nine fold-specific ridge
fits estimate gain and harm from past and path descriptors. Joint
selection under *predicted* risk passes **0/24** and trails OLS4 in
the direct conservative comparison by **11.025 pp**. Sixteen clips
violate the all-positive-harm limit even on known labels alone.
The result supports improving the risk signal and support prediction,
not immediately training a larger curved-path predictor.

~~~sh
.venv-pytorch/bin/python scripts/audit_m3w_recipient_release_capacity.py score
.venv-pytorch/bin/python scripts/audit_m3w_recipient_release_controls.py
.venv-pytorch/bin/python scripts/audit_m3w_recipient_release_amplitude.py
.venv-pytorch/bin/python scripts/audit_m3w_recipient_release_support.py
.venv-pytorch/bin/python scripts/run_m3w_recipient_release_crossfit.py
~~~

Use `replay` on the first command and `--replay` on controls,
amplitude and crossfit to verify completed local artifacts without
rerunning their optimization or fitting. Native arm64; main solve
and fixed-total control took about 85 s and 458 s, respectively,
with per-recording checkpoints and heartbeat. Eighteen scoped tests
passed. Paths and witness assignments remain local. No deployment,
new independent data, Stage5C or SMC. Native image pixels and
annotation indices, not metric, physical time or physical safety.
[Analysis](analysis.json) | [Fixed-total controls](controls.json) |
[Amplitude](amplitude.json) | [Support](support.json).
