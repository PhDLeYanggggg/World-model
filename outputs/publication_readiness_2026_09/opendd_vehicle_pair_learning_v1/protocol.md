# Learn pair risk and value directly on OpenDD

Keep the two-edit study's seven location groups, seeds 17/29/43, 8/12
native steps, matched queries and radii. Each held/calibration direction
reuses the same five-source forecaster, excluding both locations.
On each of the five risk-training donors, use the prior three-exclusion
producer that also leaves out that donor. Do not train on saved
held outcomes or on risk predictions made in-sample by another model.

In each eligible donor recording, enumerate legal distinct-actor pairs
at the same past-defined fixed query. Draw up to 256 uniformly without
replacement, using the registered seed and donor index, without looking
at labels. Supervision has four targets: the original joint risk pass
indicator (including positive easy reference), that indicator plus
positive conservative gain over both CV and OLS4, and two conservative
incremental gains relative to leaving both actors at CV. The OLS4
comparison incorporates the full recording's CV-vs-OLS4 offset only
in the binary dual label. Scale continuous labels by the two action
radii times horizon and the fixed query weight. No future validity,
risk label, true easy mass or OLS4 outcome enters the model inputs.

Train two heads per exclusion/seed for 200 updates each, with the same
training rows, two BCE labels and Huber gain loss. The additive head
averages outputs from a shared per-action network. The joint head
reads a symmetric summary of both action embeddings and six geometric
features: relative past velocity, difference in baseline motion,
difference in proposed corrections and three associated size or
alignment measures. No absolute positions or reconstructed collisions
are claimed. All heads rank by the smaller predicted gain minus 0.25
times the sum of negative log risk/dual pass probabilities. The weight
is fixed prospectively, not selected using held outcomes.

Compare learned additive, learned joint restricted to a single
family for both edited actors, and unrestricted learned joint.
Also replay the archived heuristic joint. All four arms use the
same two edits, query, paths, radii, raw label mask and actual
scorer. The fixed-count primary is direct paired learned joint vs
learned additive; a shared-family control tests the value of the
extra choices. Report both conservative trajectory and relative
motion, plus matched pass count and results after release. Each
arm calibrates on the other excluded site using the unchanged
tie-aware longest feasible score prefix. Freeze before held scoring.
Average seeds and directions within location before bootstrap.
This is exposed development, not independent confirmation.
No Songdo, reserved DroneCrowd, Stage5C, SMC or deployment change.
