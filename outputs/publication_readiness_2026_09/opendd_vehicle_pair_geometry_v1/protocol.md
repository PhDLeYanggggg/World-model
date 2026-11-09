# Observed spacing, permuted spacing, and motion-only

Test the value of actual inter-vehicle position, not another forecast
family. Reuse the seven OpenDD location groups, three seeds, 8/12
steps, fixed queries, two equal-radius edits, producer exclusions,
and direct pair training targets. Keep the same donor sampling seed,
200-update budget, optimizer and original all/easy/signed 2% checks.
All three heads share the same architecture and initial parameters.

* Motion-only: the six newly allocated channels are zero.
* Permuted: assign the observed last positions to a seeded permutation
  of the same query's vehicles. Keep each actor's actual motion history
  and candidate paths. The permutation depends only on source-local
  query roster and seed; no label controls it.
* Geometry: use the correct observed last positions for the two actors.

The additional channels encode current distance, minimum observed
distance, closing direction, constant-velocity time and distance at
closest approach (clipped to 12 steps), and minimum separation of the
two proposed paths. Compute only from observed positions, past motion
and causal forecasts. No map lookup, future coordinate or valid label
enters a feature. Positions are in a shared query-relative frame and
size channels are scaled by observed motion. Permutation is a negative
control, not a deployable policy. The zero arm retains the same parameter
count but zero channels; the permuted arm also activates those channels.

Require the geometry export to match all cohort hashes and row counts.
Reuse the earlier full recording risk labels and source-excluded
producer checkpoints. Both new and old features are attached to the
same sampled pair rows; verify that the old feature values agree.
The previously frozen learned additive policy is replayed as a
strong reference; do not retrain or tune it.

The main matched contrast is geometry minus motion-only; test
geometry minus permutation to check whether correct alignment matters,
and report geometry minus archived learned additive. Use direct paired
conservative gain, same-query relative motion, risk/dual pass counts,
and the previous recording-site bootstrap. Calibrate each new arm on
the other excluded location using the original all-recordings-feasible
score prefix, freeze before held scoring and report releases separately.
This is a controlled OpenDD development ablation, not independent
external confirmation or a claim about physical safety.
