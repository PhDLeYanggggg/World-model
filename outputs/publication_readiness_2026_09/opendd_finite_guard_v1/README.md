# Finite-harm continuation of the CV-tangent forecaster

**Decision: no policy promotion.** The finite-harm loss recovers most
of the risk-pass shortfall relative to the frozen tangent forecaster,
but loses conservative accuracy relative to the earlier shape.
Adding a curvature term does not recover additional passes.

## Matched experiment

For each of seven held OpenDD locations and three seeds, three arms
continue the same tangent checkpoint for 400 updates, with matched
AdamW state, fresh identical sample streams and donor exclusions.
All arms preserve the same 30,075 recipients and exact per-row peak
departure at the primary half radius. The prior pair-excluded
producers are reused unchanged. This is development only.

- **Continued:** original finite and tangent loss, unchanged.
- **Finite guard:** extra penalty on positive observed harm above
  the same-radius incumbent, emphasizing easy rows with donor
  recording reference normalization. That normalization is fit-only
  and bounded for training; scoring still uses the original rule.
- **Curvature guard:** finite guard plus excess finite-minus-tangent
  error over the incumbent, only where easy finite harm is higher.

| Half-radius policy | Conservative gain vs CV (pp) | Passes / 606 | Defined easy-ratio violations | Worst defined signed easy degradation |
|---|---:|---:|---:|---:|
| Incumbent | +0.219 | 547 | 29 | 8.73% |
| Earlier learned shape | +0.255 | 542 | 34 | 11.54% |
| Frozen tangent | +0.306 | 522 | 54 | 26.26% |
| Continued tangent | +0.303 | 520 | 56 | 26.00% |
| Finite guard | +0.226 | 543 | 33 | 12.42% |
| Curvature guard | +0.225 | 543 | 33 | 12.33% |

Both guards recover 21 views lost by the frozen tangent without
losing any it passed. The direct conservative finite-minus-continued
contrast is **-0.0927 pp** (95% paired location CI [-0.1272, -0.0633]);
versus the earlier shape it is **-0.0453 pp** [-0.0931, -0.0144].
Curvature minus finite is **-0.0016 pp** [-0.0027, -0.0009] with
equal pass counts. At the predetermined quarter-radius diagnostic
the guards pass 562/606, fewer than both earlier controls (564 and
566). These are direct pairwise lower comparisons, not subtraction
of two separately debited scores. Intervals average three actual
seeds within each location before 3,000 paired resamples.

## Same-radius capacity

At fixed recipients and per-row radii, selected CV reference masses
and the missing-label peak debit do not change. Positive observed harm
and signed easy degradation are monotonic in observed ADE. Thus
choosing the available path with the smallest observed ADE per row
simultaneously minimizes the original risk terms. The calculation
uses future labels and is a risk optimum, not a learned policy or
necessarily the maximum full-grid utility.

| Radius | Original bank maximum passes | Expanded bank maximum passes | No easy reference | Remaining defined failures in expanded bank |
|---|---:|---:|---:|---:|
| Half | 549/606 | 551/606 | 30 | 25 |
| Quarter | 566/606 | 566/606 | 30 | 10 |

The expanded bank adds the finite and curvature paths. Its half-radius
risk optimum gains +0.113 pp over frozen tangent under the full-grid
lower comparison, but no selector confined to these options and this
matched budget can pass every view. The result separates missing
reference support from genuinely excessive harm. Neither should be
fixed by reclassifying an undefined ratio or tuning another cutoff.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_finite_guard.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_finite_guard.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_opendd_finite_guard_capacity.py
```

63 native arm64 Torch continuations, 25,200 updates, four threads and
zero loader workers. Fresh run 28.82 s, no-fit replay 2.88 s. Fifty-four
targeted tests passed. No external readout, changed data role, risk
threshold, deployment, Stage5C, or SMC. All coordinate and sampling
claims remain in the existing OpenDD development protocol.

[Paired evaluation](analysis.json) | [Menu capacity](capacity.json) |
[Working supplement v51](../evidence_manuscript_v51/README.md).
