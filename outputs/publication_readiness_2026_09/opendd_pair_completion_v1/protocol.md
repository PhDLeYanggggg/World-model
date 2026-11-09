# Separate observed utility from the missing-step debit

Hold fixed the seven exposed OpenDD vehicle locations, three seeds,
two-edit slots, forecasts, native 8/12 grid, candidate radii and
original 2% risk checks. For each of the original 63 excluded-site
pairs and seeds, keep the same five donors, cross-fit producers,
uniformly sampled candidate pairs, optimizer and 200 updates.
Each held or calibration location is excluded from both models.

Both refits have the same symmetric pair network and five outputs.
Both predict the original risk-pass and risk-plus-dual-pass labels.
The direct control predicts both conservative incremental gains plus
an observed-gain auxiliary. The factored version predicts observed
gain, missing-step disagreement against CV, and the signed incremental
missing-step debit against OLS4 relative to leaving the actor at CV.
The observed incremental gain is identical under the two references.
Subtract the corresponding predicted debit to recover the original
ranking gains. Clip the CV debit at zero only at inference; the OLS4
incremental debit is allowed either sign. No future mask or outcome is
an inference input. The old risk logit penalties are unchanged.

Compare factored vs equal-capacity direct first; original additive
and joint are secondary controls. Report the direct paired
conservative trajectory and relative-motion differences at the same
budget, plus dual passes, missing easy support and observed utility.
Use 3,000 paired draws across the seven source means. Separately,
fit the unchanged tie-complete dual-passing score threshold on the
opposite held-out calibration location. Freeze all choices before
held scoring. This is development, not external confirmation.
No Songdo or DroneCrowd labels, deployment, Stage5C or SMC.
