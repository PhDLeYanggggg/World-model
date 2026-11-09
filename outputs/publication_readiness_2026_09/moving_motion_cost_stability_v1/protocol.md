# Stability of Motion Cost Across Fitting Sources

I test whether a forecast edit whose predicted benefit survives deletion of
any one fitting source is preferable to one ranked higher by the mean estimate.
This is a sensitivity test, not a confidence interval on a prediction and not
a deployment threshold. Earlier missing-label prediction and predictor-seed
dispersion controls tested different signals.

- Freeze the full-fit pooled cost checkpoint and its chosen motion direction
  on each row, for each of the original three seeds. Refit the same cost head
  nine times per seed, leaving out one of the nine fitting groups each time.
  Each refit uses 1,000 updates, identical architecture and optimizer, and
  site-balanced sampling over the remaining eight sources. No refit changes
  which trajectory each row would receive.
- At each recording/query, sort eligible rows by their past-only peak radius
  and pair adjacent rows without overlap. Discard pairs with a larger/smaller
  radius ratio above two. In each pair shrink both candidate edits to the
  smaller radius, without changing their directions. Keep one pair per query:
  the pair with the highest maximum full-fit predicted gain times its shared
  radius. Scores were fitted at the original radii; scaling is a causal proxy,
  not a guarantee of equal cost response under shrinking.
- Each arm edits one of the same two rows and at the same peak displacement:
  full-fit score, mean leave-one-source-out score, or minimum of those scores.
  All other rows use OLS4. Ties resolve in fixed row order after radius sort.
  Unsupported queries use OLS4, without identity certification. This forced
  experiment isolates recipient selection at a budget of one edit per query.
- Primary contrast: minimum minus mean, paired full-grid lower gain with CV
  error mass as denominator. Secondary: observed ADE, comparison to the
  original full fit and unedited OLS4, positive harm, missing-step disagreement,
  and original 2% CV-referenced recording safety. Source bootstrap: 3,000 draws
  over 14 groups after averaging three seeds. No threshold is fitted on A.
- All 34 exposed A recordings remain. Predictions and choices freeze before
  the scoring phase accesses outcomes. Nine fitting groups are disjoint from
  A. No independent confirmation, Stage5C, SMC, deployment or metric claim.
