# Cap-Event Auxiliary Cost Head

## Intended Use
Source-development research on baseline-relative expected forecasting error.
Not a trajectory forecaster, world simulator, calibrated policy or physical
safety model. Its two learned outputs are all-harm H and positive-CV-easy
harm H_E, with0 <= H_E <= H <= causal forecast disagreement. Frozen D/D_E
components are preserved for comparable downstream diagnostics.

## Inputs and Training
399 causal features:383 native features, seven context summaries, seven
missing indicators and two nested-producer risk fractions. No classifier
predictions are stacked; no future labels or actual errors are inference
inputs. Fitting-only imputation/scaling, CPU4/interop1/workers0, SiLU32,
12,899 parameters,2000 fixed updates, three seeds and all source assignments.
Three matched arms differ only in disabled/true/locality-shuffled auxiliary
event loss. Event probability is not multiplied into either expected cost.

## Limitations
Source-development localities have prior exposure. Three seeds and four-
locality bootstrap do not replace independent confirmation. Two-locality
inner versus three-locality outer risk-producer transport remains. Positive-
CV easy labels exclude perfect-CV cases; their separate guard stays fixed.
Full/motion comparisons change forecast pairs and targets, so they cannot
identify a scene/interaction/JEPA feature effect. The frozen original has
different inputs/objective; only new arms are strictly matched controls.

## Observed Outcome
All432 heads completed. Full-input easy-harm cost intervals versus cost-only:
2 favorable/0 adverse/4 overlap; versus original:0/3/3; versus shuffled:1/1/4.
All registered cost-contribution gates fail. Ranking improves versus new
controls but not consistently versus the original strong cost model. The
model is not promoted. Results are in results.md and conclusions.md; final
replay status belongs to verification.json and is not an independent retrain.
Obs8/pred12 native annotation steps and detector pixels; no metric/seconds,
human-gold, physical safety, true3D or foundation-model interpretation.
No deployment change, no independent-role access, Stage5C/SMC disabled.
