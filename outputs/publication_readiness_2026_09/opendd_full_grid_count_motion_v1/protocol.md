# OpenDD: full-grid risk at the same realized count and movement

Use the frozen OpenDD count-and-motion experiment: 202 development
recordings, seven locations, three seeds, 606 repeated views, and 780
realized edits. The causal token rules, nominated paths, exact count per
recording and online cap on the sum of peak departures from CV stay fixed.
The search can reallocate the existing prefixes over queries, not add
new paths or spend more. It is a hindsight analysis, not a controller.

Intersect the original risk and conservative two-reference constraints
with the complete-grid constraints in v211. Each selected prefix adds
its nominees' all-edited and easy excess, and one or more mandatory easy
edits must be selected. These are additive linear constraints; there
is no need to impute a future. Maximize the smaller lower gain against
CV and OLS4 using the original observed error mass. Use the original
1e-6 percentage-point solver margin and verify every witness against the
original scorer and the full-grid calculation. For any remaining numeric
or timed-out view, enumerate and re-score every legal plan if the simple
combinatorial upper bound is at most 25,000. Otherwise leave it unresolved.
Check both original unresolved views instead of assuming they are failures.

Replay the original 210 verified count-and-motion witnesses, reusing
each witness that also passes the full-grid checks. Search new allocations
where needed. Compare with the 175 verified full-grid results under at
most one exchange. Report verified lower and upper counts (the upper
adds unresolved views), and paired location intervals after averaging
seeds. No new outcomes, model updates or changed release rule.
