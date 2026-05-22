# Stage 13 Failure Analysis

Stage 13 deterministic repair did not pass the long-horizon or hard/failure gates.

## Main Failures

1. `eth_ucy_ewap` t+100 was not evaluable under the Stage 13 per-agent causal mask. This blocks any claim of pedestrian t+100 improvement.
2. HardBench improvement was only 0.013127, below the required 10%.
3. BaselineFailureBench improvement was only 0.013127, below the required 10%.
4. Interaction variants did not beat no-interaction variants.
5. Scene/goal variants did not produce a reliable lift over no-scene/no-goal variants.

## Root Cause Hypotheses

- The Stage 12 long-horizon episode builder marks source-level t+100 availability, but the per-agent visibility mask is too sparse for Stage 13 t+100 target evaluation.
- Strong causal baselines remain hard to beat, especially when residual correction is not conditioned on reliable failure detection.
- Silver/rule scene goals are not strong enough to explain long-horizon route choice.

## Required Fix

Do not enter Stage 5C. Rebuild or audit long-horizon per-agent masks first, then rerun deterministic repair before any latent/stochastic work.

