# Next Discriminating Experiment and Research Gap

This turn trained and evaluated the registered intercept repair. It made
experimental progress but did not improve a deployable policy. The overall
goal remains active, not complete, blocked or submission-ready.

## Next Minimal Question
Does the repaired head contain useful ranking information whose expected-harm
magnitude can be learned from honest fitting-only out-of-fold predictions,
or does the ranking/magnitude discrepancy persist under the same readout?

Before any new training, register a matched test of identity versus a simple
bounded nonnegative magnitude readout on both repaired-auxiliary and cost-only
scores. All readout fitting must use inner out-of-fold predictions from the
outer fitting localities, with producer, preprocessing, easy definition and
target lineage excluded consistently. Do not fit a correction on the exposed
outer scores. Preserve the existing primary and harm/tail/coverage guards.
Compare against identically processed cost-only, not an uncalibrated weak
control. This proposal is not_run and cannot be called risk calibration or
an independent validation result. Check the existing failed frozen/fractional
readouts first; do not silently repeat them or reuse their selected endpoints.

If honest inner support is insufficient, record that condition and measure
the support requirement. If matched cost-only gains equally, retain the simpler
control; do not claim auxiliary contribution. If the magnitude gap persists,
stop treating initialization or generic auxiliary classification as the main
repair and revisit conditional severity targets/data support.

## Still Missing for the Main Method
Reliable expected gain/harm costs; matched independent-agent, scene-uniform and
scene-joint policies; easy/worst-scene and interaction guards; independent
calibration and final confirmation after model freeze; strong public baselines;
reproducible scientific contribution and a complete paper/tutorial package.
Engineering fidelity and source-development ranking cannot replace these.

No new architecture sweep, threshold search, independent-role access or model
promotion is authorized by this negative result. The next experiment remains
within the existing research authorization. Stage5C and SMC stay off.
