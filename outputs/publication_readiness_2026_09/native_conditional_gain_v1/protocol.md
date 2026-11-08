# Conditional forecast gain with frozen label support

The previous box features improved missing-step prediction but did
not establish better trajectory interventions. Freeze its held-source
box support probabilities, candidates, quarter amplitude, observation
window and two-change budget. Fit only conditional gain, comparing
per-step masked regression to the same regression plus a pairwise
ranking term. Pairs are from the same query and compare both
agents only on their jointly labeled steps. This avoids using
unequal availability as evidence of which forecast is better.
Pairs with no common labels make no ranking contribution.

No probability produced by a model that saw the held source may enter
training. Train the gain heads without support probabilities, on
eight fitting sources per fold. Use the corresponding already-frozen
held-source support vector at inference and debit every unavailable
step in the evaluator. Keep the full row roster, without making
complete labels an eligibility criterion.

Run three seeds, nine held fitting sources, 500 updates per matched
arm with identical query-pair draws. At scoring compare the learned
heads against the original shared support/gain model, support debit
alone, the minimum-change policy, donor-average conditional gain,
OLS4 and same-nomination hindsight. Primary comparisons use exactly
the same counts in each within-query quartile of mean predicted
availability. The support-only policy sets those counts using its
two highest scores, before seeing any held labels. Also report the
ordinary count-matched comparison with no support strata. Equal
quartile counts do not imply equal realized availability.

Main contrasts: ranked gain versus support only, pointwise and OLS4.
Use direct full-grid paired lower scores, available ADE/FDE, hard
contribution, easy harm versus the original CV reference and OLS4
diagnostic, realized availability and matched regret. Use 3,000
source resamples after averaging seeds. Report held conditional
regression error separately. Hindsight is capacity only.

These are fitting-source experiments, not calibration or confirmation.
No threshold selection, no change to the original 2% safety rule, no
new data role, deployment, Stage5C, SMC, metric or seconds claim.
