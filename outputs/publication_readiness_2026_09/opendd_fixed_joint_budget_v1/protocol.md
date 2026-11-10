# Match the movement budget as well as the release count

In the fixed-joint OpenDD comparison, each recording's two forecasts
are unchanged, but different records can spend different amounts of
movement from CV. Test whether that difference explains the lower
conservative gain of support-weighted release. Keep the original 126
directions, seven held locations, three seeds, eight observed and twelve
forecast steps, frozen proposal scores and top-quarter count.

For each held direction, choose the same number of recordings as the
original score. Maximize the sum of the already frozen support-weighted
scores subject to spending 99% to 101% of the original mask's budget,
separately for unweighted and query-weighted displacement. Displacement
is the sum of stepwise Euclidean distances from CV over both edited
actors; the query weights use only the observed query roster. Apply the
same constraints to the support-only diagnostic. The original mask is
feasible. Recheck the chosen count and both budgets independently of
the solver; require an optimal solution in all directions. Freeze masks
without reading held outcomes. Unconstrained top-quarter masks are
replayed as a check and to show how much movement they spend.

Score the original and matched masks with the same labels, CV fallback
and 2% risk checks as before. Report conservative and observed paired
trajectory gain, relative-motion gain, and risks. Use the same 3,000
location draws after averaging directions within each held location.
The difference between observed and conservative paired gain is the
missing-label debit of the **direct** contrast. It can stay large even
when the two arms spend the same total movement, because they may
intervene on different recordings or at steps without labels.

This is a retrospective budget-matched sensitivity analysis on exposed
development data. The quota and movement budget use the full score
list, not an online policy. No training, policy promotion, calibration
change, new independent labels or new external claim.
