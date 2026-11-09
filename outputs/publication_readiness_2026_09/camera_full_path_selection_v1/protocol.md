# Past-only selection from full camera forecasts

This development experiment uses only the exposed WEPDTOF pilot for
camera scoring. Keep the original CV risk reference and easy cutoffs.
Do not use the other 13 WEPDTOF groups or DroneCrowd confirmation.

For each of seven OpenDD donor sites and three seeds, fit clean and
noise-trained splines excluding that site. Use 800 paired updates
with the same donor rows, initialization and loss. The value head
fits only on these held-site forecast banks and OpenDD labels; the
camera candidates are the previously frozen all-donor models.
The head predicts two gains, positive harms and reference masses.
Train on both the raw and equal-radius actions. No tuning on
camera outcomes or claim of independent OpenDD evaluation.

Compare nested analytic (OLS4 and median slope) and expanded banks
(add the two splines). Freeze all outputs before camera scoring.

* Raw paths: choose at most two edits per query, retain a CV agent,
  require nonpositive predicted all/easy excess at 2%, and abstain if
  no option has positive predicted two-reference gain after the pair
  penalty. This gate is a prediction, not a risk certificate.
* Matched diagnostic: per query, use the median of the larger
  analytic peak among nonstationary action rows. Shorten but do not
  enlarge any candidate. Fix the count to the smaller of two, all
  but one recipient, and the largest available single-family count.
  Compare joint, independent, common family, no pair penalty and
  independent gain-only at the same counts and total peak within
  each menu. A forced edit is not a deployment rule. Menus may
  have different supported counts, so comparisons across menus
  do not isolate the effect of adding a family.

Report all online rows including unavailable outcomes, original
all/easy/signed risk, utility against both CV and OLS4, and direct
matched comparisons. Empty matched comparisons remain not
estimable. The three seeds share one camera, not three test sites.
