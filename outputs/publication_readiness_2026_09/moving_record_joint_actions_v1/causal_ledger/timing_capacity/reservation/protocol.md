# One-Slot Causal Reservation

The frozen greedy banking policy never defers an action. This separate
exposed-development experiment deliberately defers at most one earned
switch at a time, without any lookahead or new threshold fitting. With
two slots minted and none held, compare the best feasible pair and
single. If the single and the marginal second slot both have positive
predicted utility, use the single and record the lost marginal utility
as a reservation price. At subsequent queries with capacity, spend the
saved slot only if its additional predicted utility exceeds that price.
Otherwise follow the current minted count. No retroactive edits.

Use the same frozen candidate paths, cost heads, direct query mint and
chronological predicted budgets. Scores and decision masks are unchanged
inputs. All actions freeze before this run's scoring. No use of final
recording counts in inference. The final count differs from direct by
at most one leftover slot per view. Compare to the direct control on
all 102 views; include original support, signed easy violations, number
reserved/released, and the whole-source paired completion-lower interval.
Also report exactly how many views retain equal totals. No independent
outcomes, model promotion, Stage5C or SMC.
