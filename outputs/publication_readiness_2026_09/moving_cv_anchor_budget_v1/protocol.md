# Does a CV starting forecast have enough edit capacity?

The preceding result rules out one edit per query when all other tracks start
at OLS4 under CV-referenced risk. Here every unedited track starts at CV.
I compare the exact utility capacity of the five existing analytic alternatives
(OLS4, OLS8, damped OLS4, stable turn, stationary) with a relaxed upper bound
that can make arbitrary trajectory edits. All are hindsight calculations, not
policies, training, thresholds, or a new independent test.

- Use all 34 exposed A recordings, eight observed and 12 native future steps.
  Evaluate caps 0, 1, 2, 4, 8, 16, 32, and unlimited rows per query, with one
  replacement per row. The same common known CV mass and query row weights
  used in earlier paired full-grid scores remain fixed. No test threshold.
- For each reference, compare known error savings and debit prediction
  disagreement separately at every missing step. Keep CV when no edit helps
  this lower utility. Sort positive per-row maximum improvements within each
  query; their top-k sum is exact for that objective and finite menu.
- The arbitrary path bound grants zero error at known steps and zero
  disagreement against the comparison reference at missing steps. It is
  exact for this deliberately unrestricted relaxation and an upper bound for
  every real path. It concerns the lower-bound utility functional, not an
  estimate of actual unavailable labels. Different references can choose
  different paths. Positive separate bounds do not imply joint feasibility.
- Tighten the finite-menu bound with a fixed nonnegative dual penalty on
  the original all/easy positive-harm budget excesses. Minimize over a fixed
  finite multiplier grid, never use multipliers to deploy. Drop signed easy
  and selected-reference support only in this optimistic upper bound. The
  separate realized witness at cap one uses the unchanged original scorer,
  including undefined support, and must beat both CV and OLS4 to count.
- Evaluate at source level by merging query-weighted masses per source and
  taking 3,000 paired source-group bootstrap draws. Analytic paths have no
  training seeds. No external outcomes, metrics/seconds claim, deployment,
  Stage5C, SMC, or submission status change.
