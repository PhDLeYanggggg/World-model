# Frozen Allocation And Solver Accounting

Only frozen choice matrices and allocator metadata are read. Counts concern
past-supported target agents, not all visible agents. No target labels, accuracy
or observed safety enter this report. Repeated row views are not independent.
This is not a new per-query proof or solver rerun. Nonoptimal or numerical
postcheck cases retain the feasible independent parent; do not call every query
a globally optimal joint solution.

|Lambda|Policy|Selected / repeated target views|Switch %|Nonoptimal queries|
|---:|---|---:|---:|---:|
|1|independent|1284914/4333833|29.648443|N/A|
|1|frame_uniform|209209/4333833|4.827343|N/A|
|1|joint_matched|1284914/4333833|29.648443|670|
|1|joint_free|2127712/4333833|49.095385|1206|
|2|independent|951804/4333833|21.962175|N/A|
|2|frame_uniform|99423/4333833|2.294112|N/A|
|2|joint_matched|951804/4333833|21.962175|593|
|2|joint_free|1639371/4333833|37.827277|1434|
|4|independent|679507/4333833|15.679123|N/A|
|4|frame_uniform|53371/4333833|1.231496|N/A|
|4|joint_matched|679507/4333833|15.679123|618|
|4|joint_free|1192079/4333833|27.506344|1565|
|8|independent|466936/4333833|10.774204|N/A|
|8|frame_uniform|33067/4333833|0.762997|N/A|
|8|joint_matched|466936/4333833|10.774204|553|
|8|joint_free|827679/4333833|19.098082|1750|
|16|independent|303817/4333833|7.010353|N/A|
|16|frame_uniform|22464/4333833|0.518340|N/A|
|16|joint_matched|303817/4333833|7.010353|521|
|16|joint_free|542952/4333833|12.528217|1801|
|32|independent|185726/4333833|4.285490|N/A|
|32|frame_uniform|16237/4333833|0.374657|N/A|
|32|joint_matched|185726/4333833|4.285490|481|
|32|joint_free|330375/4333833|7.623159|1778|
|64|independent|108332/4333833|2.499681|N/A|
|64|frame_uniform|12389/4333833|0.285867|N/A|
|64|joint_matched|108332/4333833|2.499681|372|
|64|joint_free|184752/4333833|4.263016|1546|
|128|independent|61515/4333833|1.419413|N/A|
|128|frame_uniform|10050/4333833|0.231896|N/A|
|128|joint_matched|61515/4333833|1.419413|235|
|128|joint_free|98176/4333833|2.265339|1263|
|256|independent|35362/4333833|0.815952|N/A|
|256|frame_uniform|8362/4333833|0.192947|N/A|
|256|joint_matched|35362/4333833|0.815952|111|
|256|joint_free|52758/4333833|1.217352|868|
|512|independent|21397/4333833|0.493720|N/A|
|512|frame_uniform|6611/4333833|0.152544|N/A|
|512|joint_matched|21397/4333833|0.493720|79|
|512|joint_free|30779/4333833|0.710203|518|
|1024|independent|13683/4333833|0.315725|N/A|
|1024|frame_uniform|4621/4333833|0.106626|N/A|
|1024|joint_matched|13683/4333833|0.315725|50|
|1024|joint_free|19706/4333833|0.454701|313|
|2048|independent|8992/4333833|0.207484|N/A|
|2048|frame_uniform|3375/4333833|0.077876|N/A|
|2048|joint_matched|8992/4333833|0.207484|51|
|2048|joint_free|13109/4333833|0.302481|233|
|4096|independent|5938/4333833|0.137015|N/A|
|4096|frame_uniform|2165/4333833|0.049956|N/A|
|4096|joint_matched|5938/4333833|0.137015|37|
|4096|joint_free|8914/4333833|0.205684|146|

N/A means no MILP for that arm, not a measured zero solver error.
Exact fast paths, successful MILP solves and distinct parent-fallback reasons
are retained separately in the JSON. Joint allocation pools expected error
budgets; it does not optimize physical pairwise conflicts. Actual risk still
requires the separately frozen calibration readout.
