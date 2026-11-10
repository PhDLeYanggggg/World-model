# Full-grid easy risk sensitivity

Use the same 160 previously scored Songdo confirmation flights, three
seeds, original 2% checks, published easy cutoff, and two-edit packets.
Do not fit, change the release decisions, or open another cohort. Compare
the resulting diagnostic with the original risk checks and the earlier
full-grid all-edited sensitivity. Never count a missing reference as a pass.

Let B be the sum of the observed CV errors for an edited row, H the observed
sum of proposed-minus-CV errors plus the distance between the forecasts at
each missing step, and r = 0.02. These are twelve-step sums, so the published
easy cutoff applies to B/12. For an easy row, its maximum positive-harm
excess over the limit is max(H, 0) - r B. Placing missing targets at CV
attains the largest signed harm and the smallest baseline denominator. If
B = 0 on an incomplete row, use a limit of arbitrarily small positive CV
error to keep the row in the strictly positive easy subset.

A complete positive easy row must be included. An incomplete row is
optionally easy when B/12 is at most the cutoff: it can instead be made
non-easy by increasing one missing point's distance to CV. Other rows are
excluded. The exact supremum of the threshold excess is the sum of the
mandatory terms plus the positive parts of the optional terms. This is over
free completions per scored occurrence, not physically feasible paths. A
complete positive easy edit is required for every completion to have a
selected-easy denominator. The two uncertainty modes (missing denominator
or excess above 0) can overlap because they can have different witnesses.

For a complete grid with a positive selected-easy denominator, the signed
easy degradation on the whole roster is at most the positive selected-easy
ratio: unedited errors cancel from the numerator and can only add mass to
the denominator. So checking all-edited risk and selected-easy support and
risk suffices for all three numerical checks on every completion. This does
not relax the original empirical release rule or establish physical safety.
