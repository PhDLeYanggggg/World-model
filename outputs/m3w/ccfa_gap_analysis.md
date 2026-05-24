# M3W CCF-A Gap Analysis

## Current Verdict

not yet CCF-A candidate.

## What Is Missing

1. Model gap: M3W hybrid has not exceeded Stage26 on t+50 or hard/failure.
2. Representation gap: hybrid beats Transformer-only in this run, but JEPA latent non-collapse is still below gate and the contribution needs retrained ablations.
3. Experiment gap: current ablations are inference-time; retrained ablations are still needed.
4. Statistics gap: bootstrap CI exists, but multi-seed variance is still needed for stronger claims.
5. Data gap: SDD is pixel-space raw-frame only; effective seconds/homography/metric scale are unverified.

## Shortest Path

1. Treat Stage26 selector as the deployment floor and train M3W latent features only as auxiliary selector features.
2. Run retrained ablations with no-scene/no-goal/no-interaction/no-JEPA/no-Transformer under the arm64 torch runtime.
3. Add multi-seed or larger bootstrap evidence and per-scene/per-agent-type breakdowns.

## Usable Paper Material

- Strict leakage-free SDD pixel-space benchmark setup.
- Strong Stage26 cost-aware selector baseline.
- Runtime-safe M3W JEPA/Transformer implementation and negative results.

## Claims That Must Not Be Made

- Do not claim true 3D.
- Do not claim metric trajectory prediction.
- Do not claim Stage5C or SMC readiness.
- Do not claim M3W beats Stage26.
- Current M3W t+50 improvement is `0.1308150291442871`, below Stage26.
