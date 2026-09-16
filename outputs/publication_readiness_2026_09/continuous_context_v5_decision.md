# Continuous-Identity Development Context Repair

The fit-only public predictor experiment uncovered an upstream context issue,
before its development scoring. The [row audit](students01_packaging_audit/audit.md)
maps all 17,820 packaged Students01 rows to the same timestamp and rounded
coordinate in the continuous 21,813-row local annotation file. It also proves
exact 20-point truncation: 415 physical tracks become 891 chunk IDs, all 63
tracks shorter than 20 points disappear, and 3,993 tail/short-track rows are lost.
Chunk retention depends on later availability. Past-only access to that packaged
cache does not establish causality of the upstream observation population.

The v4 runner is halted after preserving the current complete full-forecaster
checkpoint. This is a source-integrity correction, not a response to low model
accuracy, slow training or development-based threshold selection. There has been
no v4 development scoring. The old runs and immutable source snapshots remain.

## Repair Frozen Before New Scoring

- Rebuild Students01 only from the already-local continuous `students001.txt`,
  preserving original IDs, timestamps and source coordinate precision.
- Keep short tracks and all known past rows in the context store. Future path
  completeness controls label eligibility only, never the inference population.
- Keep the same physical-scene role: historically exposed University development,
  not untouched test or an additional scene. Students03 remains the same group.
- Preserve every fit recording, fold, seed, model, loss, 10,000-update budget,
  easy/hard definition, graph geometry, reference floor and policy threshold.
- Retrain under a new version rather than relabel old checkpoints with a new
  protocol hash. The completed old fit remains available for exact old-source
  replay; it is not discarded or represented as a completed three-seed study.
- Run the EqMotion fixed-head adaptation and matched local Transformer on the
  repaired observation universe. Report all registered seeds and failed controls.

This repairs one demonstrated packaging mechanism, not every upstream annotation
operation. The continuous file's annotation interpolation and physical/time
calibration remain unverified; it is not certified sensor-as-of ground truth.
The task is native-grid obs8/pred12, raw t+50 separate, dataset-local units only.
The repaired development row count differs from v1-v4: do not interpret a
before/after aggregate difference as a model-only gain. One physical development
site still cannot support an independent-scene CI. Stage5C and SMC remain off.
