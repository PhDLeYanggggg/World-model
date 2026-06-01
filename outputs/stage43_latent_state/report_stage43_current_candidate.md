# Stage43 Current Integrated Candidate

- source: `fresh_stage43_aq_integrated_candidate_gate`
- result_source: `fresh_integrated_manifest_from_stage43_aj_to_ap_artifacts`
- verdict: `stage43_aq_integrated_protected_latent_state_candidate_pass`
- gate: `14 / 14`
- policy hash: `4dc482d146af2940b4968385e3f4bdf6951036b27b0353f1759a2130265ed493`
- current candidate supported: `True`
- long objective complete: `False`

## Current Best Deployable

- name: frozen Stage43 bounded-residual policy under Stage37/teacher safety floor
- deployable: `True`
- global floor removed: `False`
- h100 guarded: `True`
- why: Exact reviewer replay, positive bootstrap deltas over stored hard-switch policy, zero easy degradation, and explicit t100 raw-frame guard.

## Metrics

- all improvement vs floor: `38.00%`
- endpoint improvement vs floor: `44.21%`
- t50 full-waypoint improvement vs floor: `26.96%`
- t50 endpoint improvement vs floor: `35.59%`
- t100 raw-frame diagnostic vs floor: `0.00%`
- hard/failure improvement vs floor: `37.71%`
- easy degradation vs floor: `0.00%`
- switch rate: `63.08%`
- reviewer replay max abs diff: `0.00000000`

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
