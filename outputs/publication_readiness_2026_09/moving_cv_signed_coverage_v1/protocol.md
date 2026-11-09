# Direct and Conditional Signed Risk With Query-Swap Gain

Hold the frozen causal candidate bank, query-local pooling algorithm,
target grid and original recording rule fixed. Train on analytic
proposals only inside the previous disjoint train/calibration/held
source rotations. Held paths include source-excluded donors. Use the
same causal motion and CV-edit features as the frozen admission head.

Fit two equal risk heads for 1,000 updates each with identical
initialization and draws. Direct predicts the two signed excesses
without factorization. Conditional predicts signed all excess given
some future support and signed easy excess given easy reference;
combine conditional means with the same three-class support task
and the exact causal radius for wholly missing rows. Both regressions
are robust location estimates, not guaranteed conditional means.
The 2% term, support definitions and future masks are used only to
construct training and evaluation targets, not inference features.

Fit a third gain head for 1,000 updates. It predicts the two original
CV-anchored gain terms and adds a weighted preference between edits
on different agents in the same query. A preference only contributes
when its known-step advantage exceeds the missing-step path debit
for both edited agents. Compare with the frozen gain head under each
of the three risk readouts. The gain comparison changes both the fit
and the objective; it is not a matched loss-only ablation.

Report the six factorial diagnostics at the same fixed predicted
screen, with donor recovery on the same true action and every
recording risk pass retained. Compare new-risk/old-gain pools against
gain-only controls with exactly their per-query counts. Differences
among other arms can change coverage. Report source-bootstrap
intervals (3,000 draws after averaging seeds), CV/OLS4 utility,
neighbor displacement, and tail easy risk. Calibration considers
the six fixed arms, margins 0/.005/.02 and minimum predicted easy
probability .10/.25 with no artificial query cap. Require all
calibration recordings to pass the original rule and positive
conservative source-macro gain over both references, or return CV.
All held actions freeze before scoring. No A outcome, deployment,
independent confirmation or change to the original 2% rule.
