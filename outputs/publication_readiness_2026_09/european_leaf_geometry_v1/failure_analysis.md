# Failure Analysis

The refit averages B/e, H/e and EH/e over known training rows in each existing
leaf, then multiplies by the causal query envelope. It preserves reference
predictions and feature partitions. This changes target parameterization and
the effective regression objective; it is not an equal-loss neural experiment
or a calibrated upper bound.

1. Scale mixing is measurable, but its direct correction fails. Almost all
   additions (17,719/17,733) have a training-leaf mean envelope above the query
   envelope. Aggregate leaf/query envelope mass is 3.44 for the added pool.
   This describes the neighborhood, not a uniquely identified causal mechanism.
2. Reference outputs are bitwise frozen, unlike earlier fractional neural
   auxiliaries that could degrade a shared reference head. Refitting that head
   cannot explain this failure; selection still changes its aggregate denominator.
3. Changed decisions are worse targeted. Known harmful additions total 5,591,
   versus 1,907 known harmful removals. Added unknowns are 247, removed unknowns
   170. Total intervention falls, but unknown exposure rises from 918 to 995.
4. Known-label easy-risk violations rise 4->21. Another 13 refit failures arise
   only under unknown completion. Do not label unknowns harmless or use future
   availability as an inference filter.
5. The largest utility declines are locality008 (-0.13993%) and110 (-0.11235%),
   normalized by full known reference mass. Largest signed-MSE increases are
   082 (+0.46436) and067 (+0.39576). A pooled scalar does not locate every failure.
6. Matched support improves 29->30, but matched utility has a negative point
   estimate and overlapping interval. Selecting only that favorable support
   count would hide failure of the registered comparison.

The experiment does not identify how much error comes from detector-silver
trajectory/identity quality, genuine but causally ambiguous motion, or routing
optimized for the previous absolute targets. It does not exhaust jointly trained
relative-target forests, but provides no evidence to scale them up now.

The earlier observation-quality audit left sparse event-label quality untested.
Connect current event-bearing rows to that lineage evidence before changing
another objective; quality proxies do not establish human-verified truth.

Only exposed source development was used. No independent scenes, metric/seconds,
human-gold, physical safety, true3D or foundation claims. Stage5C/SMC remain off.
