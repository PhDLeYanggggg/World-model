# Pre-Readout Amendment: Keep the Support Difference Explicit

The registered v1 attempt stopped as designed, before new metric evaluation.
PID52141 exited1 on25September at the common-support guard after15 partial
decision groups; no complete decision manifest or outcome analysis was produced.
The original code, registration and partial archives remain unchanged.

A causal-score-only audit of all36 groups identifies two mismatches, both
damping easy-event groups in fold1. Seed17 has31 eligible rows whose product
reference estimate is exactly zero; hurdle switches27 of them. Seed29 has36
such rows and hurdle switches3. These are view-specific counts, not independent
examples. The hurdle estimates remain positive. The audit uses no new future
outcomes and does not establish the numerical cause of the zero estimates.
See [causal support receipt](causal_support_mismatch.json).

The original question needs an explicit support component. Silently dropping
these rows would hide part of the original policy difference. Assigning an
arbitrary finite score to0/0 would change ranking semantics. Therefore, before
new readout, retain six views in each of36 groups:

1. Original product policy on its full legal pool.
2. Original hurdle policy on its full legal pool.
3. Product anchor restricted to the common causal pool.
4. Hurdle anchor restricted to the common causal pool.
5. Hurdle ranking at the common product count, per locality.
6. Product ranking at the common hurdle count, per locality.

The two full originals must still exactly reproduce the parent72 views. All
216 views are reported. Count matching is only within the common causal pool;
counts are frozen without future-label filtering. Both original support strata
are retained in full-anchor error accounting. For each locality and subset:

```text
full total = support difference + common-pool total
common-pool total = ranking-at-product-count + coverage-with-hurdle-ranking
                 = coverage-with-product-ranking + ranking-at-hurdle-count
```

Every component uses the same CV error-sum denominator. Undefined localities
remain undefined. This is accounting, not a unique causal attribution. The
2% rule, labels, forecasts, source roles, seeds, bootstrap and tie-breaking do
not change. No thresholds are tuned. Matched counterfactuals remain offline
diagnostics and can violate the risk rule. No model or deployment is selected.

The amended runner/module/config and this file are frozen in a second commit
before amended decisions/readout. Outputs and logs are isolated under
`support_v2/`, preserving the stopped attempt. New training remains zero.
Independent roles stay closed; no Stage5C/SMC or metric/seconds/physical-safety
claim. Released detector-track pixels,8/12 rawstride12 development only.
