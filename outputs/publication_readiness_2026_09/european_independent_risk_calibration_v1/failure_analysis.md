# Calibration Limitations And Failure Analysis

Complete dedicated-role calibration, not independent confirmation. No model,
policy, risk budget or threshold grid is changed by this diagnosis.

No common multiplier passes the original full-record contract. Global-floor fallback is separate, not learned safe success.

## Observed Error Harm Versus Support

|Lambda|Policy|Observed-harm exclusive|Undefined reference|Unknown-bound only|Supported|No actual switches|
|---:|---|---:|---:|---:|---:|---:|
|1|independent|899|9|4|6|3|
|1|joint_matched|907|9|2|0|3|
|2|independent|881|9|6|22|7|
|2|joint_matched|902|9|1|6|7|
|4|independent|859|10|11|38|8|
|4|joint_matched|882|10|5|21|8|
|8|independent|841|15|14|48|9|
|8|joint_matched|863|16|9|30|9|
|16|independent|795|23|26|74|12|
|16|joint_matched|814|23|17|64|12|
|32|independent|730|43|32|113|25|
|32|joint_matched|754|43|22|99|25|
|64|independent|642|80|46|150|53|
|64|joint_matched|675|80|37|126|53|
|128|independent|505|146|51|216|118|
|128|joint_matched|539|146|41|192|118|
|256|independent|402|221|40|255|188|
|256|joint_matched|425|222|42|229|188|
|512|independent|298|316|25|279|280|
|512|joint_matched|312|315|26|265|280|
|1024|independent|205|426|21|266|400|
|1024|joint_matched|219|426|19|254|400|
|2048|independent|136|567|9|206|544|
|2048|joint_matched|138|567|12|201|544|
|4096|independent|94|677|5|142|656|
|4096|joint_matched|96|675|6|141|656|

Exclusive categories prioritize undefined reference before observed harm;
overlapping flags are retained in risk_failure_taxonomy.json. Harm here means
increased forecast error, not physical damage. Empty selected reference is
not a measured safe zero. Full-roster easy-reference unavailable views:
0/918.

The low-multiplier failures are not explained solely by unobserved outcomes.
Increasing the multiplier suppresses actions, but finite selected support is
lost and average gains approach zero. Passing signed easy net alone would
not pass the separate original positive-harm/reference constraints.

## Selected Cost Diagnosis

This attribution was added after the completed readout, not preregistered as
another primary result. It uses native costs, never lambda-scaled optimizer
coefficients. Medians condition on defined record ratios; nine pipelines
repeat recordings and do not create independent samples.

|Lambda|Policy|Median predicted/true H|Median predicted/true EH|
|---:|---|---:|---:|
|1|independent|0.201091|2.96084e-05|
|1|joint_matched|0.195606|3.10265e-05|
|4096|independent|0.0108455|2.92595e-14|
|4096|joint_matched|0.0103443|3.63045e-14|

The cost accounting separates harm underprediction from reference
overprediction; it does not identify the causal training mechanism. A new
SOURCE-side repair should address the selected easy-harm tail and verify
SOURCE generalization, not tune repeatedly on these calibration labels.
Future reuse of this cohort must disclose its outcome exposure.

The complete paired policy effects are in paired_calibration_effects.md.
Some grid points show positive descriptive effects, but inspecting the grid
does not supply multiplicity-controlled superiority or repair failed risk.
No favorable lambda is substituted for the registered unsupported selection.

Available-label ADE is not full12 ADE. Unknown-row completion bounds do not
recover missing steps inside partial trajectories. Pixels/raw stride12 and
released tracker/detector silver do not support metric, seconds, human-gold,
true3D or foundation claims. Confirmation and DroneCrowd remain closed.
Stage5C and SMC remain off. No new controller is deployed.
