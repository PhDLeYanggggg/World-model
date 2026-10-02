# Why Deleting Unstable Decisions Failed

## Not Just Too Few Recordings

For raw-pool calibration, the source-count decomposition is exact:

| Source recordings | Groups | Parent selected | Consensus selected | Complete pass old -> new |
|---|---:|---:|---:|---:|
|2|48|10,692|0|15 -> 0|
|3|12|60|60|3 -> 3|
|17|6|65,726|65,726|5 -> 5|
|19|6|5|5|0 -> 0|

Two-recording groups cannot support the nested deletion. But the remaining
groups retain exactly the same actions, including every original upper-risk
failure. More available recordings therefore do not by themselves explain away
the error. For example, the19-recording source074 retains the two fully observed
3.1129% easy-harm views. Its other retained view fails positive utility despite
an easy ratio below2%. Risk is not the only condition for complete support.

The primary's360 changed deletion applications include newly admitted actions
as margins loosen; the final consensus intersects those actions with the parent.
Counting any decision change without its direction would overstate the rule's
ability to reject harmful parent actions. It rejects no supported parent action.

## Unknown Support Versus Observed Harm

The secondary selected-set consensus has three nonempty risk failures:

| Source/controller/head | Retained | Unknown | Observed easy positive harm | Completion upper |
|---|---:|---:|---:|---:|
|067/controller1/head17|2|1|0.0000%|31.8705%|
|067/controller1/head29|7|2|2.1801%|44.2450%|
|008/controller0/head43|4|0|7.7296%|7.7296%|

The first is unsupported because of the unknown completion, not an observed
31.9% error. The second has both observed excess harm and unknown support. The
third is entirely observed and cannot be explained by missing future labels.
Its17-recording source still produces a harmful retained set after deletion
consensus. Unknown labels are never converted to harmless outcomes.

## What This Does and Does Not Establish

- It rejects this fixed nested-deletion consensus as a useful repair, under the
  original models, masks, source roles and2%budget.
- It shows that support loss accounts for the primary's entire intervention
  reduction. Lower coverage must not be credited as confidence discrimination.
- It does not establish that all recording-stability measures are useless,
  that conditional outcomes are intrinsically unlearnable, or that collecting
  additional recordings would solve the remaining observed-harm errors.
- It does not isolate domain shift, label noise or a single feature defect.
- It supplies no reason to retune deletion votes, quantiles or target thresholds
  on these results. There is no new deployment or world-dynamics contribution.

Prior causal-support factorization also lost useful interventions; prior head
seed changes left persistent risk failures. This control adds a different
perturbation with the same practical warning: rejecting more rows is not an
alternative to predicting their incremental harm and benefit accurately.
