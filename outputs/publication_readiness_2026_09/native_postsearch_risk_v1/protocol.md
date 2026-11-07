# Source-Nested Risk Estimation After Joint Search

Test whether a risk estimator trained on **selected** search actions
improves the development policy over a predicted risk-margin control.
This uses the same bank, recipient count, query cap, exact joint
objective and 2% scoring rule as the prior path portfolio, but refits
all action-cost and shortlist coefficients within separated sources.
It is a new matched development comparison, not a replay of its scores.

For each forecaster seed and held source, use the same registered
seven fitting and six donor sources from the earlier source-role
design. Inside the seven fitting sources, generate choices leaving
the chosen source out of every shortlist and cost fit. On these
cross-fitted choices, predict two events: nonpositive combined
all/easy risk excess, and presence of observed easy reference. Inputs
are predicted pre-search costs, path disagreement, graph score,
scale, crowd size and number of alternatives. No future feature.

Refit search coefficients on the seven sources and use those fixed
coefficients on all six donors and the held source. Donor outcomes
choose among a fixed ladder of seven thresholds based on training
score quantiles, including the ungated search-admitted control.
Rank nonempty operating candidates by donor recording passes, then
total-risk violations plus undefined easy views, then conservative
OLS4 utility, then smaller intervention count. The learned score
uses predicted safe probability times (0.5 + 0.5 easy-support
probability). The control uses minimum predicted all/easy slack.
If no donor threshold passes every recording, report that explicitly;
the selected operating point is exploratory, not admissible.

Primary contrast: the learned decision vs. a retrospective
risk-margin rank at identical intervention counts per recording
and action cardinality. This matched control is not an online
policy. Report the unfiltered screen-first and donor-selected
margin policy separately. Score 102 held development views (34
recordings, 14 sources, three forecaster seeds), with 3,000 paired
source draws after seed averaging. Report observed calibration
Brier score where search found options, actual risk passes and
CV/OLS4 conservative utility. Eight observed / twelve predicted
annotation steps in image pixels. No external role access, new
deployment, Stage5C or SMC.
