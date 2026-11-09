# Factor easy support, then calibrate on an unseen source

Keep the existing seven OpenDD location folds, 8/12 native samples,
and original recording 2% rules. Rotate all six calibration locations
for each outer held location; this is the established nested development
design, not a new official split. The same five-site vehicle forecast
excludes the two locations from both calibration and held inference.
When fitting the risk models, each donor action uses a forecast trained
without that donor and without either of the two excluded sites. The
35 unique triples and three seeds give 105 producer fits. Both risk
models see the same five donor batches, paths and easy cutoff (donor
complete CV errors only) with no target or calibration data in fitting.

The control estimates gains, harm and reference mass directly. The
factorized arm estimates the probability of a usable easy reference,
easy positive harm and reference mass conditional on support, plus
unavailable-label debit separately. Its predicted easy numerator is
`p(support) * conditional_harm + unknown_debit`; its denominator is
`p(support) * conditional_mass`. Conditional training uses only true
support rows. The other targets and training budget are common.

For this mechanism test use one shared slot per recording, at the query
with the largest number of legal distinct actors (first in timeline on
a tie). This uses only past-defined candidates and fixes exact count,
query and radius for both risk models. Rank each action by minimum
incremental two-reference gain minus positive predicted all/easy
excess. Use the top action in this slot as the matched diagnostic,
even if it would not be released. The scope is a one-edit risk test,
not a claim about joint interaction selection.

On the independent calibration site, order its proposed edits by the
same score. Choose the longest score-threshold prefix where **every**
edited recording passes the original all/easy/signed 2% rules and has
positive conservative gains over CV and OLS4. Tied scores move together.
No qualifying prefix means CV fallback. Freeze all directions before
held scoring. Report the matched ranking and calibrated policy separately.
Average repeated calibration directions and seeds within outer site
before bootstrapping locations (3,000 draws). Do not interpret these
rotations as independent recording samples or a physical safety claim.
No Songdo or confirmation labels, threshold tuning on held outcomes,
deployment, Stage5C or SMC.
