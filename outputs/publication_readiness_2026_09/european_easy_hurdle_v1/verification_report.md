# Verification Scope

All 108 action groups and the complete development readout replay exactly.
Independent arithmetic checks cover 4,487,400 query constraints,
2,376 policy cost views, 432 probability/cost quality views,
and 4,346 locality reductions. Matched policies differ in 30,402
queries despite exactly equal per-query counts. These counts repeat role and
seed contexts and are not independent sample sizes.

All 51 scoped tests in 10 files pass. All 216 restored checkpoint
hashes agree with the committed CREATE manifest. The first pair's 4,000 training
updates replay exactly on the same CREATE runtime except elapsed time; the
other 214 heads were verified, not retrained. Local arm64 action and evaluation
replays do not prove cross-architecture training equivalence.

The full legacy integration suite and a cold raw-data rebuild are not_run.
The former contains unrelated training/report-writing workflows; a scoped pass
is not described as a whole-repository pass. These engineering checks confirm
the negative result's arithmetic and provenance, not scientific efficacy.

The exploratory screen still fails. Independent confirmation remains unopened;
no deployment change, metric/seconds claim or safety certificate. Stage5C and
SMC remain disabled. verification.json binds source and public artifact hashes.
