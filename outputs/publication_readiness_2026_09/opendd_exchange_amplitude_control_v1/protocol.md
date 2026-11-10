# OpenDD: match the cumulative displacement of the late shape

At the same frozen 101 exchanges, compare the original residual,
its late phase warp, and a non-warped amplitude control. The control
uniformly contracts the residual at mutable steps if the late
shape uses less cumulative deviation, and uniformly fills a fraction
of headroom at those steps if it uses more. Preserve direction at
each step, exact peak steps, terminal position, and the same total
norm as the late phase path. No future positions are used for this
matching; inability to match any chosen row stops the experiment.

Compare both against original and directly against each other
using the original risk, utility and paired location summaries.
The phase and amplitude controls must not change the action roster,
selected rows, count, movement cap or threshold.
