# Original versus final selected-score errors

## Material Passport

fresh_run:all72 groups of critic inference and label-aware arithmetic,plus a separate exact complete diagnostic replay.
cached_verified:four frozen critic families,source packets and original/final policy choices. No choice is reoptimized.
Run PID62326: 133.040s;diagnostic replay PID64645: 131.791s.
All288 original-set MSE controls match existing readouts within registered floating tolerance.
This replays diagnosis,not the optimizer;it is not a replacement for either complete policy replay.
No model,threshold,decision rule,risk criterion or data role changes.

## Query support and changed selections

|Critic / selected set|Selected row views|Complete queries|Unknown queries|Empty queries|Introduced rows|Changed actions|
|---|---:|---:|---:|---:|---:|---:|
|new_row_decoded|429205|70363|6770|35463|216418|129349|
|new_row_original|265671|58970|3294|50332|0|0|
|new_set_decoded|410575|64820|6471|41305|200594|126725|
|new_set_original|265671|58970|3294|50332|0|0|
|old_row_decoded|440480|69490|6549|36557|199222|144921|
|old_row_original|265671|58970|3294|50332|0|0|
|old_set_decoded|432796|67449|6157|38990|179777|133154|
|old_set_original|265671|58970|3294|50332|0|0|

Complete means every selected row has five available ADE-derived cost labels,not that every future point or endpoint exists.
Counts are reused context/head-seed views,not independent agents or scenes.
All causal rows remain. Unknown/empty means are not imputed as zero.

## Conditional score accounting

Scores use the same causal-scale units,not different training-loss denominators.
Average supported context/seed views within locality,then supported localities.
Support differs by final selection;these conditional means are NOT full-cohort or causal effects.

|Critic / set|Channel|Score MSE|Mean underprediction|Harm component|Reference component|Supported views/localities|Pred<=0,true>0 queries|Zero true references|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|new_row_decoded|all|0.00417977046|0.00889741587|0.00797366956|0.00092374631|71/72;12/12|23791|0|
|new_row_decoded|easy|0.000193681388|0.00173291922|0.00177975338|-4.68341558e-05|71/72;12/12|13134|25945|
|new_row_original|all|0.000674088075|0.00235035186|-0.00142873165|0.00377908351|66/72;11/12|7306|0|
|new_row_original|easy|0.000102786391|0.000671707446|0.000868436235|-0.00019672879|66/72;11/12|3576|13596|
|new_set_decoded|all|0.00356266381|0.00570272222|0.00450681225|0.00119590997|70/72;12/12|21037|0|
|new_set_decoded|easy|0.000154541727|0.00161570375|0.00165935848|-4.36547252e-05|70/72;12/12|12574|22278|
|new_set_original|all|0.000648157535|0.00263768243|-0.00114140107|0.00377908351|66/72;11/12|7553|0|
|new_set_original|easy|9.93231665e-05|0.000884236434|0.00108096522|-0.00019672879|66/72;11/12|3799|13596|
|old_row_decoded|all|0.0030195882|0.00845808682|0.00719515337|0.00126293345|70/72;12/12|29030|0|
|old_row_decoded|easy|0.000479885284|0.00500326502|0.00506585952|-6.2594498e-05|70/72;12/12|24271|21836|
|old_row_original|all|0.0010431787|0.00815381173|0.00437472822|0.00377908351|66/72;11/12|16900|0|
|old_row_original|easy|0.000190911244|0.00316938681|0.0033661156|-0.00019672879|66/72;11/12|14394|13596|
|old_set_decoded|all|0.00265728939|0.00680800404|0.00530006655|0.0015079375|70/72;12/12|27908|0|
|old_set_decoded|easy|0.000474467905|0.00491501969|0.00499957046|-8.45507698e-05|70/72;12/12|24867|18942|
|old_set_original|all|0.000974965658|0.0071616714|0.00338258789|0.00377908351|66/72;11/12|16958|0|
|old_set_original|easy|0.000193034001|0.00334182592|0.00353855471|-0.00019672879|66/72;11/12|18255|13596|

The identity is: true-minus-pred signed score = true-minus-pred harm plus
0.02 times pred-minus-true reference,each divided by summed causal scale.
Positive reference contribution means reference inflation also understates the signed risk.
This is arithmetic,not a unique causal attribution or conditional-mean misspecification proof.
Earlier candidate-invariant reference sharing also failed its matched study;reference bias here
does not establish incoherence as a sufficient explanation or a new successful repair.
Strict sign counts are not the ratio-support/finite-record2% gate;zero reference remains undefined.
All original risk requirements remain in the main complete policy readouts.

## Interpretation boundary

Use this audit to distinguish original-set fitting from final-selection errors before the next model change.
A better original-set MSE alone is neither final-selection calibration nor deployability.
No new inferential CI is claimed for these descriptive conditional diagnostics;primary policy intervals
are reported in the corresponding whole-roster results,including negative and unknown-completion outcomes.
Exposed SOURCE fitting only,not whole-pipeline OOF or independent confirmation.
Task:obs8/pred12 at raw stride12,detector-silver/local-image coordinates.
No new trajectory dynamics,metric,seconds,human-gold,true3D,foundation,submission-ready claim.
Deployment unchanged;independent roles,Stage5C execution and SMC stay closed.
