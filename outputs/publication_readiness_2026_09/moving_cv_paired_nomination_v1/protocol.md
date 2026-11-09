# Conservative Within-Agent Preference Loss

The preceding head already predicted two CV-anchored gains, including the
increment of the OLS4-relative lower bound. Merely adding an OLS4 label would
repeat that objective. Test an auxiliary preference loss that compares two
candidate forecasts directly, before choosing among the legal trajectories.

For candidate A versus B, sum the distance-to-label advantage of A on
valid future steps and subtract the sum of the distances between A and B
on missing steps. A positive result implies a definite preference under
any completion of the missing labels. Do the same in reverse; if neither
lower bound is positive, the pair supplies no preference supervision.
The preference loss is softplus of the signed score difference divided by
0.15, weighted by the guaranteed margin divided by history speed and
forecast length (clipped at one), and added to pointwise loss at weight
0.25. Both arms receive identical causal inputs, including displacements
from both CV and OLS4. Their pointwise targets are the two original
CV-anchored incremental bounds. The control has preference weight zero.

Same nine fitting/calibration/held cells as the preceding experiment.
Train two heads per cell, 1,000 AdamW updates, identical initialization,
matched draw stream and analytic-only training/calibration paths. Held
paths include source-excluded neural donors. CV-risk/easy eligibility is
determined by the unchanged frozen head, not by the new rank score.
The primary comparison uses the prior fixed diagnostic risk screen and
exact previous counts within each query. Include the preceding full-menu
screen result as a matched reference. Compare the paired objective to
the pointwise control with paired full-grid conservative utility and
3,000 source bootstrap draws after seed averaging. Separately count
definite path wins/losses where both arms intervene on the same agent
but choose different trajectories; those rows are not independent trials.

Run the unchanged 15-setting calibration on each new ranker, requiring
the original 2% rule in every view and positive lower utility versus both
references. If no setting qualifies, return CV without claiming edited
support. Hold the original rule, primary task, 15/30 DroneCrowd design
and all sealed outcomes fixed. Neither fitting-source readout nor replay
is independent confirmation.
