# Tighter Cost Support: Limited SOURCE Improvement

The zero-update mechanism check completed on all nine original mean_absolute
SOURCE heads. Runtime was 117.301 seconds, peak RSS 10,552,279,040 bytes.
Original prediction hashes match all nine separately verified SOURCE controls.
No neural weights, forecasts, thresholds or policies were trained or changed.

| Target arithmetic | Original mean decoded objective | Projected objective |
|---|---:|---:|
| Exact native labels, normalized by frozen causal scale | 0.721499640 | 0.718860046 |
| Original float32 training targets | 0.721499644 | 0.718860051 |

The table averages the nine complete producer objectives. Each retains the
original query visits and observed-agent/legal-action weighting; unknown
queries remain zero-supervision rather than fabricated labels. The objective
combines five RMS-scaled cost errors and half-weighted absolute-gain error.
It is not uniformly pooled bank MSE, validation utility or forecast error.

All native feasible five-cost distances are nonincreasing, as expected for
classical weighted projection. The complete objective also happens to decrease
in all nine cases, by about 0.37% on average. The gain-term improvement is an
observation here, not a projection guarantee. R and ER predictions are exactly
unchanged. Native and float32-target arithmetic are reported separately.

The reduction is small. On the same native query-weighted metric, reference
cost MSE remains 1.24971209 and easy-reference MSE 0.32099488. EH MSE changes
only from 0.40285849 to 0.40270400. These numbers are not comparable to the
earlier uniformly pooled action MSE without its different weighting/RMS basis.
This does not establish that the loose bound is the primary failure cause, or
that a tighter decoder trained end-to-end would fix selection and risk.

Four regression checks include comparison with an independent constrained
optimizer, feasible-label distance, idempotence and invalid-input rejection.
These are engineering/numerical checks, not learned-policy success.

[Evidence figure](figures/SOURCE_cost_support.svg) is generated only from public
scalar receipts. Its three panels and captions were inspected; two independent
SVG/PNG renders and receipts match exactly. The combined scoped suite covering
projection, support, original controller, cardinal cost and forest checks passes
29 tests in 5.32 seconds. The full repository suite was not run in this step.

No development, private calibration, confirmation or DroneCrowd arrays read.
No selector policy is evaluated in this diagnostic; no deployment promotion.
The pending complete forest experiment retains its original primary. This
remains local-image-pixel/raw-frame/silver fitting evidence, not metric,
seconds, full-12-step error, physical safety, true 3D or foundation capability.
Stage5C and SMC remain off.
