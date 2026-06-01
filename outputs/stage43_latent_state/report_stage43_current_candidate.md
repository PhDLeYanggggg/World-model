# Stage43 Current Integrated Candidate

- source: `fresh_stage43_aq_integrated_candidate_gate`
- result_source: `fresh_integrated_manifest_from_stage43_aj_to_ap_plus_stage43_p_artifacts`
- verdict: `stage43_aq_integrated_protected_latent_state_candidate_pass`
- gate: `18 / 18`
- policy hash: `03497313f878a1ec69fd7d2824842fee0acfa79c38dc9d667c6d6ac53ef4c331`
- frozen replayable policy hash: `4dc482d146af2940b4968385e3f4bdf6951036b27b0353f1759a2130265ed493`
- current candidate supported: `True`
- long objective complete: `False`

## Current Best Deployable

- name: Stage43-P protected tail-horizon full-waypoint adapter under Stage37/teacher safety floor
- deployable: `True`
- global floor removed: `False`
- h100 guarded: `True`
- why: Validation-selected tail-horizon full-waypoint adapter improves full-test all/t50/hard over the frozen bounded-residual replay, preserves easy cases, and explicitly floors unsafe h100/t100 switches.

## Frozen Replayable Safety Artifact

- name: frozen Stage43 bounded-residual policy under Stage37/teacher safety floor
- policy hash: `4dc482d146af2940b4968385e3f4bdf6951036b27b0353f1759a2130265ed493`
- reviewer replay max abs diff: `0.00000000`
- all improvement vs floor: `38.00%`
- t50 full-waypoint improvement vs floor: `26.96%`
- hard/failure improvement vs floor: `37.71%`
- easy degradation vs floor: `0.00%`

## Metrics

- all improvement vs floor: `50.25%`
- endpoint improvement vs floor: `51.15%`
- t50 full-waypoint improvement vs floor: `51.23%`
- t50 endpoint improvement vs floor: `55.13%`
- t100 raw-frame diagnostic vs floor: `0.00%`
- hard/failure improvement vs floor: `47.88%`
- easy degradation vs floor: `0.00%`
- switch rate: `70.45%`
- reviewer replay max abs diff: `0.00000000`

## Latest Tail Adapter Delta Vs Frozen Replay

- all delta: `12.25%`
- t50 delta: `24.27%`
- t100 delta: `0.00%`
- hard/failure delta: `10.16%`
- easy degradation delta: `0.00%`

## Domain Deltas Vs Stored Hard Switch

| domain | delta |
| --- | ---: |
| ETH_UCY | `8.21%` |
| TrajNet | `15.02%` |
| UCY | `3.78%` |

## Horizon Deltas Vs Stored Hard Switch

| horizon | delta |
| --- | ---: |
| 10 | `2.75%` |
| 25 | `7.81%` |
| 50 | `10.52%` |
| 100 | `17.79%` |

## What This Does Not Claim

- Not true 3D.
- Not foundation-scale.
- Not metric or seconds-level.
- Not global safety-floor removal.
- Not Stage5C execution.
- Not SMC.

## Why The Long Goal Remains Active

- The long objective still asks for broader multimodal latent world-state evidence, more source calibration, and final full-suite replay.
- The current candidate remains protected and floor-dependent.
- Metric/time calibration and true 3D evidence remain unavailable.
- Stage5C and SMC remain disabled.
