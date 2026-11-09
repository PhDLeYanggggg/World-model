# Imitation of verified full-path plans

Use the existing disjoint 8-to-12 OpenDD development windows.
On each training recording solve the unchanged CV risk and
two-reference maximin problem, with the full action bank and
at most two edits per query. Retain at least one CV agent.
Use only verified positive plans or certified nonpositive or
infeasible instances; unsupported and unresolved instances are
not negative training labels. Construct each training site's easy
cut from that site's training labels. At held-site scoring use
the positive complete-case quartiles of the fitting sites only.

Learn query slates of zero, one or two actions. Both the additive
and the pair-coupled score rank the demonstrated slate against
the same sampled alternatives. The pair residual takes last
positions and relative changes of the proposed paths, not true
future positions. At inference enumerate all available slates,
including abstention, with no threshold search.

1. Use OLS4 and median slope for the held-OpenDD location
   experiment. Train on other locations only, at three seeds.
2. Use those two paths plus the previously frozen clean and noisy
   splines for the camera experiment. Teacher paths for each
   donor site come from forecasters excluding that site. Score
   only the already exposed Convenience Store camera. No camera
   labels for fitting, and no scoring of the 13 external groups.

Compare CV, OLS4, independent and joint. Matched replay gives
the independent and common-family controls the joint plan's exact
peak slots and edit count in each query. A path may be shortened,
never enlarged, to fill a slot. These controls use the same
forecast directions but their realized paths can be shortened.
If a common family cannot fill the exact slots, mark that
recording's common-family contrast unavailable. No empty control
counts as a zero effect or a risk pass.

The teacher uses donor labels; it is not a policy or a guarantee.
Held-site scores are source-excluded development evidence, not
independent confirmation. Use the original all/easy/signed 2%
rule for all scored arms. Keep all unavailable futures in the
conservative calculation. No DroneCrowd confirmation access.
