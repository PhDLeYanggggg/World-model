# Current-Selection Neural Cost Training

18 fresh neural cost heads, 36,000 primary updates, three actual seeds; same models, inputs, initializations, sampled queries and budget as frozen-teacher controls. The only loss change is detached current independent selection instead of fixed teacher selection. Both complete model banks froze before any new H readout. All 54 neural prediction arrays and all policy blocks replay; all three cached control decisions match the verified parent. Tree costs were not freshly inferred; cached allocation replay is not bitwise MILP reoptimization.

|Adaptive joint-free versus|Known utility pp [95%CI]|Direct completion-lower pp [95%CI]|
|---|---:|---:|
|point|-0.3725 [-0.7601,-0.0163]|-1.5040 [-2.8957,-0.5496]|
|shared_tree|+10.9899 [+4.8057,+17.4180]|+4.7951 [-2.8864,+11.3193]|
|frozen_teacher|+4.5480 [+1.5691,+8.1255]|+0.9663 [-1.6809,+3.4314]|

|Joint-free arm|Support /348|All violations|Easy violations|Undefined easy views|Worst signed easy-net %|
|---|---:|---:|---:|---:|---:|
|point|10|273|281|46|376.331142|
|shared_tree|37|56|67|222|0.902000|
|frozen_teacher|14|251|249|70|309.589585|
|adaptive|9|274|277|52|376.331142|

|New bank|Policy|Repeated queries|Nonoptimal/fallback queries|
|---|---|---:|---:|
|frozen_teacher|joint_matched|16734|619 (3.70%)|
|frozen_teacher|joint_free|16734|312 (1.86%)|
|adaptive|joint_matched|16734|387 (2.31%)|
|adaptive|joint_free|16734|203 (1.21%)|

Supported-source mechanism gate:False. New deployment: False. Original 24-group primary: not_estimable, not passed.

Current selection is causal but the record loss uses C supervision. Its selection is independent-agent, not the final joint choice; sampled record multisets are not unbiased full-video loss estimates. No gradient through action choice; this is not SPO+ or a new safety theorem. A no-switch batch has no extra record term, not an observed safety pass.

solver_diagnostics.json retains actual per-query reason counts for both new banks, including time/status limits and postcheck fallbacks. Feasibility replay does not mean every decision was an optimal MILP solution. No solver budget was increased.

All 348 repeated recording views, four policies, original 2% positive-harm/support and signed-net condition remains. Empty same-count utility stays NA. Known-label and bounded-completion estimands are separate. Cost ratios/zero-CVcounts are supplementary diagnostics, not model selection. The Cdiagnostic uses independent choices, while the Hdiagnostic uses joint-free choices; their difference mixes population and decoder changes, not isolated domain shift. All 23 supported groups are exposed development, not independent tests or verified IID scenes. Reserved 20 Moving groups, DUT and DroneCrowd confirmation remain closed.

No new trajectory model, metric/seconds/physical-safety/true3D/foundation/human-gold claim. Native 8/12 supplied annotation order, not verified sensor-as-of time. Stage5C/SMC off.
