# Full Source-Training Exposure and Learning-Rate Diagnostic

## Material Passport

Prospective registration, before continuation fitting. Existing source data and
three parent checkpoints are `cached_verified` only after hash/schema/role and
score replay. New optimizer updates and full-training scores are `fresh_run`.
Held-source/main forecasts, main training, development/calibration/confirmation,
deployment selection and submission readiness are `not_run`.

## Why This Experiment

The completed sixty-model cost comparison did not beat CV even on full training
sets. The subsequent twelve microfits learned selected training rows with the
original decoder despite clipping every update. The latter is memorization, not
generalization. Complete-complement training gave about8.30 draws per row;
microfit gave2000. We now test longer exposure and step-size decay without
changing data, supervision, architecture or output scale. This is a prerequisite
diagnostic for a useful gain/harm intervention predictor, not the method claim.

## Fixed Comparison

- Use all15430 complete stationary source-training rows outside bookstore;
  excluded bookstore is not scored. This is the first registered source fold,
  not a newly selected favorable evaluation domain.
- Same480 past geometry/frame features, eight cached pastRGB crops, original
  context-radius bounded decoder, ADE objective and uniform batch64 sampler.
- Fork verified `ade_past_rgb_bookstore_seed17/29/43` checkpoints at step2000.
  Preserve model, AdamW moments, sampler state, Torch RNG, feature normalization,
  training-label loss scale and draw counts. Parent files remain immutable.
- Six branches: each of three parents gets constant and cosine schedules.
  Both run8000 additional updates to global10000, total48000 new updates;
  unique inherited computation is6000 updates, not six independent fresh starts.
- Constant learning rate0.0003. Cosine decays from0.0003 to0.000003 across the
 8000 new updates, including exact endpoints. AdamW weight_decay0.0001, clip5.
  Effective AdamW shrinkage changes with rate; this is an optimizer-schedule
  comparison, not proof of a pure gradient-step mechanism.
- Evaluate the entire unchanged training complement at steps2000/4000/6000/10000.
  Save each checkpoint and training forecast. No best-step or best-seed selection.
  First100 continuation updates may be used for timing, within the fixed budget.
- Record trainADE/CV, moving/zero-target and hard diagnostics, pixel error,
  every-update clipping counts, sampled-row exposure and batch-loss trace.
  Zero-CV easy percentage remains undefined; report absolute harm instead.
- Three paired optimizer-seed ranges are descriptive, not a generalization CI.
  No bootstrap of training windows is presented as independent evidence.

## Interpretation and Next Branch

If full-training accuracy improves only with decay, step-size floor is a
supported local explanation, not yet a transfer result. Improvement under both
schedules supports budget/exposure as a factor, without proving convergence.
If both still fail, investigate input/target ambiguity, generated annotation
variation and capacity before scaling blind benchmark matrices. A successful
training fit alone cannot open calibration/confirmation or promote a model.
No negative result is discarded, no primary metric or original population is
silently redefined. The main offline supplied-history8-to12 task is unchanged.

## Resources and Verification

Local nativearm64 `.venv-pytorch`, Torch CPU4/inter-op1, workers0. Prior matched
fits suggest roughly40minutes plus full-training evaluation; pilot measures
actual cost. Atomic checkpoint every200steps, heartbeat at each checkpoint,
optimizer/sampler/RNG resume. CREATE has only a saved authentication blocker;
current remote jobs/assets are unknown, no duplicate remote submission.

Targeted tests: constant continuation versus original uninterrupted algorithm,
both-schedule exact resume, immutable completed resume, matched sampling and
parent optimizer immutability. Final checks replay all24 training predictions,
verify three paired streams and preserve all parent/milestone artifacts on
completed resume. Full legacy suite is not assumed rerun.

This remains SDD source-only, annotation-pixel/past-normalized raw-frame
diagnosis, not metric, seconds-level, true3D, foundation or submission-ready.
Generated/interpolated past annotations are not strict sensor-as-of or human
intention gold. No Stage5C execution or SMC.
