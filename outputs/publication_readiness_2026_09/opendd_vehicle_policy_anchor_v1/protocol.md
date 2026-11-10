# Align the training recipients without changing families or held actions

Train a nested copy of the deployed-study additive pair chooser.
For each triple of excluded locations, train only on the remaining
four. Candidate forecasts for each of those four training sites come
from the existing checkpoint excluding that site and all three
held locations. Derive the easy and hard cutoffs on the four fit
sites alone. Use the original additive network, candidate-pair
sampler, risk/gain supervision, merit, 200 updates and three seeds.

At each outer ranker-training recording use the first draw from
the registered family-aligned sampler. Use one example per recording
in both arms, rather than repeating the policy's top pair. For the
policy arm select the highest additive-merit pair with the **same
ordered path families** and at the same query. This is the original
additive rule conditional on family, not an unrestricted change to
the proposal bank. Use the same excluded angle nominator and
same exact success labels for each arm. Both pointwise and budget
losses use identical update counts and initializations.

At calibration and held time, keep the exact prior actor, angle and
per-step paths. Compare both new pairs of heads with each other and
with frozen original action margin. Use the same 526 and 1,368
aggregate offline budgets and the unchanged separately calibrated
2% risk and positive incremental-gain criteria. Use 3,000 paired
location draws after averaging seeds and directions.

Retain the preceding three-draw family-alignment result as a
reference, not as the primary control for this experiment. Report
the one-draw support explicitly. OpenDD is development. No new
external outcomes, release promotion, Stage5C, or SMC.
