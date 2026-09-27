# Why Predicted Risk Does Not Match Selected Harm

Post-readout diagnosis of the frozen216 linear heads. No tuning, refitting or deployment.
Each row aggregates equally over12 locality means, with dependent fits averaged first.

| Target | Predicted selected harm/reference | Actual selected harm/reference | Predicted/actual harm | Predicted/actual reference | Selected zero-harm prediction fraction | Actually harmed among zero-harm predictions |
|---|---:|---:|---:|---:|---:|---:|
| cv | 0.0043 [0.0035, 0.0051] | 0.0473 [0.0375, 0.0574] | undefined | 2.0210 [1.3072, 3.1710] | 0.6528 [0.5874, 0.7151] | 0.2222 [0.1978, 0.2486] |
| floor | 0.0043 [0.0035, 0.0051] | 0.0491 [0.0389, 0.0598] | undefined | 2.0339 [1.3125, 3.2082] | 0.6536 [0.5872, 0.7174] | 0.2389 [0.2135, 0.2659] |

Undefined means at least one fixed source/view has no valid denominator; empty coverage is not zero harm.
Ratios above are fractions, not percentage points. They concern the reference used by each target.
The main report evaluates BOTH policies relative to the actual damping floor.
This identifies moment underestimation and clipping behavior, not a fitted causal explanation or risk guarantee.
