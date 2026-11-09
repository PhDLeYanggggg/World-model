# Coherent expected harm and reference mass

Keep the original 8/12, analytic bank, CV/OLS4 comparators, 2% checks,
source roles and three seeds. No A or independent confirmation labels.

Both heads receive the same causal history/action features and share
the architecture, training draws, loss and class targets. Support
class and reference mass are predicted from history only. The direct
head predicts total all/easy positive harm (including missing debit),
signed easy change, and nested reference masses. The component head
predicts conditional observed harm/benefit and conditional reference
mass. Compose their moments with probabilities of wholly unknown,
observed easy and observed other; the missing debit is the predicted
wholly unknown probability times exact causal peak displacement.
Use identical weighted squared marginal moment loss and class loss.
Do not assume independence between easy status and observed harm.

Freeze the v141 shape model's gain estimates. Hold exact per-query
shape-portfolio counts for the primary direct/component comparison.
Also compare with the same shape risk, and each model under the
existing free-count risk-priced portfolio. Calibrate only on the
assigned calibration sources, with the original checks. After
freezing, report matched-policy risk, source-paired utility and pair
geometry, plus moment prediction error on the original portfolio's
chosen actions. This last diagnostic uses a common exposure set and
does not establish calibration or replacement-policy performance.
