# Matched upward harm refinement: complete readout

Fresh54tree refiners /1080primary tree fits on18cached-verified neural heads. No new neural training. All18original fitting outputs and held baseline outputs replay exactly; all72held cost outputs and policy scalars replay, with72saved solver ledgers. Three complete first-job refits are exact.

|Contrast, joint-free|Known effect pp [95%CI]|Completion lower pp [95%CI]|
|---|---:|---:|
|adaptive_minus_fixed|+0.0705 [+0.0140,+0.1366]|+0.0027 [-0.1412,+0.1087]|
|adaptive_region_minus_frozen|-11.3580 [-18.1372,-4.8396]|-18.5094 [-29.1914,-9.0506]|
|fixed_region_minus_frozen|-11.4285 [-18.2394,-4.8862]|-18.5226 [-29.0693,-9.1142]|
|global_minus_frozen|-11.3362 [-18.1176,-4.8264]|-18.4877 [-29.1366,-9.0204]|

|Arm|Supported views /348|All risk violations|Easy risk violations|Undefined selected-easy views|Worst known easy net %|
|---|---:|---:|---:|---:|---:|
|frozen|10|273|281|46|376.3311419298208|
|global|42|40|69|221|3.254035237092972|
|fixed_region|47|36|68|217|2.9164839440793062|
|adaptive_region|47|42|72|214|1.3706116333181426|

Registered supported-group primary positive: False. Complete original risk pass: False. Neither a positive pooled/known score nor a floor-only decision can replace the complete contract.

All6policies,138source/producer/seed views and348repeated recording views are retained. Intervals use3000source-group bootstrap resamples after producer/seed averaging, not IID windows. The original24group primary remains not estimable; this exposed23group study cannot independently recertify deployment. Empty same-count contrasts stay not estimable. Exact replay checks saved solver bindings and feasibility, not bitwise MILP reoptimization.

The registered paired effects use a common observed-label floor-error denominator. Completion bounds add worst-case path disagreement for wholly unknown rows; partially observed rows still use available-label ADE. These are not full12-step complete-trajectory ADE reductions or bounds on a ratio with an unobserved complete-data denominator. An unchanged floor has zero path change analytically, while unsupported selected-event risk remains not estimable.

[All policies and intervals](summary.json), [secondary metrics](secondary_metrics.json), [fitting moment factorization](fitting_summary.json), [switches and empty controls](policy_usage.json), [solver ledger](solver_summary.json), [actual runtime](runtime_summary.json), [decision](decision.json).

[Fixed choice-region fitting comparison](fitting_check/readout.md) separates estimation changes from abstention. [Post-decision held-cost diagnostics](held_cost_check/readout.md) retain every causal policy and recording without changing any model prediction or cached decision. These are diagnostics, not independent calibration guarantees.

No new deployment or submission readiness. Native8/12steps and dataset-local pixel coordinates only; no verified seconds, metric scale, true3D or foundation claim. Stage5C and SMC remain off.
