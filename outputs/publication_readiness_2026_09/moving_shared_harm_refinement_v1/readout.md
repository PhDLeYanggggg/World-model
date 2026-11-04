# Shared past/current context in harm refiners

36fresh435-column tree refiners /720primary trees,18cached-verified neural heads,3seeds. Both local-zero and shared controls are retrained before H.54fresh cost arrays and108choice scalar blocks replay exactly. Prior387decisions are cached-verified with fresh complete scalar replay; their costs are not freshly inferred.

|Joint-free contrast|Known effect pp [95%CI]|Direct completion-lower pp [95%CI]|
|---|---:|---:|
|local_minus_frozen|-11.3205 [-18.0986,-4.8407]|-18.4711 [-29.1042,-9.0813]|
|shared_minus_cached_adaptive_region|-0.0044 [-0.1731,+0.1781]|-0.0173 [-0.1876,+0.1639]|
|shared_minus_cached_fixed_region|+0.0660 [-0.0718,+0.2330]|-0.0042 [-0.1957,+0.1857]|
|shared_minus_cached_global|-0.0262 [-0.1997,+0.1367]|-0.0376 [-0.2128,+0.1254]|
|shared_minus_frozen|-11.3624 [-18.1160,-4.9176]|-18.5141 [-29.1153,-9.1278]|
|shared_minus_local|-0.0420 [-0.2044,+0.0931]|-0.0539 [-0.2168,+0.0815]|

|Cost/policy arm|Full support /348|Known all violations|Known easy violations|Undefined selected easy|Worst signed-net easy %|Switch %|
|---|---:|---:|---:|---:|---:|---:|
|frozen|10|273|281|46|376.331142|70.92582|
|local_zero|35|57|76|218|2.063147|5.53541|
|shared_motion|37|56|67|222|0.902000|4.56742|
|cached_global|42|40|69|221|3.254035|5.89275|
|cached_fixed_region|47|36|68|217|2.916484|4.52464|
|cached_adaptive_region|47|42|72|214|1.370612|5.33085|

Registered robust primary passed: **False**. Full original risk passed: **False**. No new deployment or submission readiness. All six policies and23sourcegroups remain in summary.json/groups. Original24primary stays NA; this is exposed development, not confirmation.

|C common choice region|Cost model|H actual/pred|EH actual/pred|
|---|---|---:|---:|
|frozen|frozen|42.701241|202.249747|
|frozen|local_zero|0.777936|0.367467|
|frozen|shared_motion|0.770253|0.351311|
|local_zero|frozen|7.001387|69.630628|
|local_zero|local_zero|1.015849|1.010054|
|local_zero|shared_motion|0.556645|0.460917|
|shared_motion|frozen|19.195756|277.962699|
|shared_motion|local_zero|0.574072|0.429019|
|shared_motion|shared_motion|0.918627|0.819859|

|H own joint choice, post-decision|H actual/pred|EH actual/pred|R actual/pred|ER actual/pred|
|---|---:|---:|---:|---:|
|frozen|30.221184|99.341989|0.777870|1.284454|
|local_zero|1.247950|1.535622|0.892651|0.931603|
|shared_motion|1.069716|1.274014|0.879161|0.891604|

Within-region comparisons hold selection fixed. Own-region C loss/masses are not H safety. Pooled costs are not source-macro accuracy or recording-level certification. Empty same-count comparisons stay NA. Signed-net preservation cannot replace positive-harm/support.

Resource phases and real durations are in runtime_summary.json. Saved choices are binding/feasibility verified where fresh costs exist; no bitwise MILP reoptimization is claimed. No new images, independent roles, metric/seconds, true3D, foundation, Stage5C or SMC.
