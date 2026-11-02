# Calibration Risk Failure Taxonomy

Descriptive readout only, not another policy or parameter search. All 102 recordings
and all 918 pipeline-record views are retained at every grid point. Exclusive
categories use the listed priority; the JSON also retains overlapping flags.
Known label coverage is reconciled across every policy and producer.

|Lambda|Policy|Unsupported|Observed harm|Easy net only|Unknown-bound only|Supported|
|---:|---|---:|---:|---:|---:|---:|
|1|frame_uniform|170|713|0|8|27|
|1|independent|9|899|0|4|6|
|1|joint_free|9|906|0|2|1|
|1|joint_matched|9|907|0|2|0|
|2|frame_uniform|242|631|0|12|33|
|2|independent|9|881|0|6|22|
|2|joint_free|9|903|0|2|4|
|2|joint_matched|9|902|0|1|6|
|4|frame_uniform|327|521|0|19|51|
|4|independent|10|859|0|11|38|
|4|joint_free|10|889|0|6|13|
|4|joint_matched|10|882|0|5|21|
|8|frame_uniform|407|404|0|22|85|
|8|independent|15|841|0|14|48|
|8|joint_free|14|868|0|13|23|
|8|joint_matched|16|863|0|9|30|
|16|frame_uniform|503|298|0|22|95|
|16|independent|23|795|0|26|74|
|16|joint_free|21|819|0|28|50|
|16|joint_matched|23|814|0|17|64|
|32|frame_uniform|596|213|0|13|96|
|32|independent|43|730|0|32|113|
|32|joint_free|40|740|0|45|93|
|32|joint_matched|43|754|0|22|99|
|64|frame_uniform|694|139|0|10|75|
|64|independent|80|642|0|46|150|
|64|joint_free|77|656|0|61|124|
|64|joint_matched|80|675|0|37|126|
|128|frame_uniform|777|70|0|9|62|
|128|independent|146|505|0|51|216|
|128|joint_free|138|525|0|52|203|
|128|joint_matched|146|539|0|41|192|
|256|frame_uniform|821|40|0|6|51|
|256|independent|221|402|0|40|255|
|256|joint_free|214|429|0|50|225|
|256|joint_matched|222|425|0|42|229|
|512|frame_uniform|865|16|0|2|35|
|512|independent|316|298|0|25|279|
|512|joint_free|309|309|0|38|262|
|512|joint_matched|315|312|0|26|265|
|1024|frame_uniform|885|5|0|1|27|
|1024|independent|426|205|0|21|266|
|1024|joint_free|423|212|0|25|258|
|1024|joint_matched|426|219|0|19|254|
|2048|frame_uniform|898|1|0|0|19|
|2048|independent|567|136|0|9|206|
|2048|joint_free|558|143|0|14|203|
|2048|joint_matched|567|138|0|12|201|
|4096|frame_uniform|903|0|0|0|15|
|4096|independent|677|94|0|5|142|
|4096|joint_free|674|93|0|6|145|
|4096|joint_matched|675|96|0|6|141|

Full-roster easy-reference unavailable views: 0/918.
This availability flag is identical across all actions and multipliers.
Where the complete known-easy reference mass is zero, no selected subset
can have positive known-easy reference under the unchanged floor, easy
definition and labels. That is a support obstruction, not something a
different threshold or trajectory predictor can repair on these same labels.

Unsupported reference mass is not a safe zero. A completion-bound-only
failure does not prove missing rows actually incur worst-case harm. Observed
positive harm and signed easy-net degradation are distinct. All figures are
conditional calibration readouts, not independent confirmation or a population
guarantee. Partial-future ADE does not recover unobserved steps.
