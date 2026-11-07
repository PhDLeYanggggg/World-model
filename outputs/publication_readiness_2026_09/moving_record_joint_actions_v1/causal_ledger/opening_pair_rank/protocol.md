# Rank Useful Opening Candidates Within A Query

The frozen bank has a positive supported candidate in 15 of the 18
zero-direct views' first two queries, but the tested top-one scores
find none. Keep the 8/12 annotation-step image-space task, seven
fitting, six calibration and one held source in each of 42 cells,
frozen trajectories, and original 2% observed risk rule.

Use an identical small shared-trunk Torch model and the same fitting
draws for both arms. The positive label is a strictly positive ADE
gain and a positive easy reference with original standalone support.
Sample queries with equal source probability, alternating an unrestricted
draw and a draw from queries with a positive. Within each, include up
to four positives, eight highest previous-score negatives, and eight
additional negatives. The previous score and all labels in this
sampling are fitting-only. Pointwise training balances positive and
negative mass in each represented query. The ranking arm adds a
within-query logistic positive-negative margin with weight 0.5.
Both get 140 updates, identical initialization and batch indices.
Model scores are not calibrated probabilities.

Primary component: first-two-query top-one and top-eight recall on the
registered 18 zero-direct views, with the full bank ceiling and existing
joint scorer. Also report first and all eligible query slices. The
slice is previously exposed and descriptive. For the all-view policy,
use the previous donor-only threshold grid and risk preservation rule.
The unthresholded first-query policies use identical queries and counts
to isolate ranking. Score all 102 views with unchanged risk, paired
completion-lower utility and whole-source intervals. No new forecaster
training, independent outcomes, deployment, Stage5C or SMC.
