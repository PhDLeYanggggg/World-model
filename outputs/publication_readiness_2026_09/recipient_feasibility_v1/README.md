# Attainable Risk Does Not Recover Matched-Budget Utility

**Fresh native Torch fit.** Two matched objectives, nine held-source
folds, three seeds, 54 heads and 27,000 updates. Both heads regress
the reference and risk implied by the best legal path at a row:
minimum ADE when observed, minimum disagreement debit when wholly
missing. The treatment also supervises observation support, a
locally safe easy anchor, and nonharm. Identical seeds, sampled rows,
architecture, past inputs and fold-local easy thresholds within each
pair. The old evaluation threshold is unchanged.

Every arm uses the same frozen source-excluded path nominator.
Only recipient ranking changes. The first row preserves the prior
supervised recipient set with this common nominator. The recent-gain
control ranks the frozen path model's predicted recent-blend gain.
Recording-wide ranking can compare later query prefixes; it is an
offline allocation diagnostic, not a live causal policy. The
query-quota arm only uses its own query's past features and quota.
Each arm uses the same 95,769 replacements.

| Recipients, common nominated path | Conservative gain vs CV (pp) | All-risk >2% | Actual risk passes | Hindsight risk passes, same recipients |
|---|---:|---:|---:|---:|
| Prior recording ranking | +1.419 | 49 | 0 | 0 |
| Recent-gain recording ranking | +2.358 | 58 | 0 | 0 |
| Minimum-risk moments, record | +3.173 | 53 | 0 | 1 |
| Auxiliary support, record | +3.547 | 54 | 0 | 1 |
| Auxiliary support, query | +1.497 | 64 | 0 | 0 |

Counts are out of 72 views (24 recordings, three seeds). The primary
direct conservative comparison to the prior recipient set is
**+1.589 pp [-2.733,+6.637]**. Auxiliary supervision against minimum
moments alone is **-0.022 pp [-0.791,+0.673]**. The direct contrast
to OLS4 is **-6.548 pp [-13.919,+0.941]** for auxiliary support at
recording total. Intervals use 3,000 paired source-group draws after
seed averaging. The one hindsight risk witness does not beat both
references. This does not establish a meaningful improvement.

## Why more search will not fix this estimated budget

For one predicted risk inequality, sorting eligible rows by
`predicted_harm - 0.02 * predicted_reference` gives its exact minimum
at a fixed count K: select the K smallest terms. If this minimum
is positive, even ignoring the other risk condition and allowing
fractional choices cannot yield a feasible selection. This easy-risk
lower bound is positive in **all 69 nonempty views**. Three views
have zero quota and remain unsupported. The check concerns the
model's optimistic attainable-risk estimates, not actual
infeasibility: the [fitting hindsight audit](../recipient_release_capacity_v1/README.md)
finds 14/24 actual witnesses at the same recording totals.

We also ran a [query-local fallback](../recipient_query_fallback_v1/README.md)
that uses the nominated path's own predicted costs and permits
fewer changes. It is not a rescue: 286 changes, 6/72 risk passes,
no utility gain or difference between the two same-count orders.
Do not deploy either arm. This result rules out just tuning the
price grid for this predictor. The next method should test whether
features available at a query can distinguish which recipients
provide reference support and low harm under a fixed path bank,
with genuinely source-excluded calibration rather than forced
recording-wide allocation.

~~~sh
.venv-pytorch/bin/python scripts/run_m3w_recipient_feasibility.py
.venv-pytorch/bin/python scripts/run_m3w_recipient_feasibility.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_recipient_relaxed_search.py
.venv-pytorch/bin/python scripts/run_m3w_recipient_query_fallback.py
~~~

Native arm64 CPU, four threads, one inter-op thread and zero
workers. Atomic checkpoint and heartbeat every 100 updates, same
RNG on resume. Choices frozen before evaluation. Exact score
replay, 24 scoped tests. All data and checkpoints stay local.
No new calibration, confirmation, deployment, Stage5C or SMC.
Native image-coordinate units and annotation steps.
[Analysis](analysis.json) | [Predicted capacity](search_audit.json).
