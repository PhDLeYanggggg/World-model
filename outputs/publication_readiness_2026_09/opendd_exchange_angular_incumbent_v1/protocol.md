# OpenDD: learn the utility of a direction against the actual incumbent

Keep all 780 edited recipients, their per-step displacement norms,
the 606-view evaluation, and the original 2% rule. Keep the angular
architecture, zero-rotation initialization, 8/12 native steps, three
seeds and 400 training updates. Draw 24 examples from each donor
location per step, identically for every arm. Donor rows are the
existing causal analytic histories at the three fixed bridge fractions;
no fitted proposal or label comes from the outer held location.

Compare the existing CV-and-OLS4 worst-reference objective, the
same objective plus mean direct conservative gain against the
unrotated edited path, and the latter with half the incumbent term
placed on the two worst donor-location batch means. Higher loss
means worse conservative gain. Use the direct path disagreement on
missing steps for each comparison. All weights and update counts
are fixed; held data never select a checkpoint or threshold.

Primary development comparisons: paired conservative utility
against the original incumbent and the matched dual-reference fit,
with dual-pass counts and 3,000 location bootstrap draws after
averaging seeds. Report observed step/endpoint errors and tails.
No policy promotion, external label access, or changes to the
existing role assignments.
