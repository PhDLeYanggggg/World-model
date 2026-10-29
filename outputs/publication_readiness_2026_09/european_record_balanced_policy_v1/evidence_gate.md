# Evidence Gate

This is a mechanism-level fitting gate, not a substitute for the full M3W or
submission-readiness criteria. No negative historical result is relabeled.

|Check|Result|Evidence|
|---|---|---|
|Real neural training|pass|18Torch fits,36,000updates,captured exit0,12.2minutes|
|Registered comparison and frozen finals|pass|Registration hash verified;7481dfd5before readout; all18finals scored|
|Recording accounting|pass|74unique source-qualified recordings;148record/context views;444with repeated headseeds|
|No new evaluation-role leakage|pass for this run|Fitting-only input filtering; independent roles unopened; inherited historical exposure not rehabilitated|
|Recording constraint coverage improves|partial|84to131of444;59gained,12lost; still313unsupported|
|Full-policy all/easy2%budgets|fail|290all-upper violations,59easy-upper violations,9undefinedeasy|
|Net easy degradation<=2%|pass on TRAIN only|0of444violations; does not waive positive-harm constraints|
|Matched-count utility vs parent|positive descriptive TRAIN bound|[+0.020396,+0.021290]pp,utilityonly,notCI|
|Full utility vs parent|negative TRAIN bound|[-0.135714,-0.133642]pp|
|All-record candidate support|fail|Known locality082/51fixed-candidate obstruction retained|
|Replay and targeted engineering tests|pass|72exactreadouts,1,836arithmeticchecks,68+3targetedtests|
|Independent predictive generalization|not_run|No new held or independent confirmation|
|Deployment promotion|false|Full-policy budgets fail|
|Submission readiness|false|Mechanism fitting result; independent evidence and method-level superiority missing|
|Stage5C execution|false|Not executed|
|SMC|false|Not enabled|

Count-matched controls are utility-only. Empty matched risk, if encountered,
remains not_estimable. The original2%requirements still apply to the complete
policy across required scenes/recordings. A fallback is not a proof of learned
neural safety, and an average pass cannot hide individual failures.
