# Frozen-Interaction Cost Head: Model And Data Card

Status: diagnostic research model, not deployed. Training and evaluation are
fresh_run; original EqMotion predictors, source roles and parent controls are
cached_verified. There is no new independent final confirmation.

## Model

- 508 inputs per candidate: 444 existing causal motion, rollout and detector
  fields, plus 64 frozen ego interaction features. The matched control sets the
  final standardized block to exactly zero. Both heads have 33,349 parameters.
- Seven candidate actions and five cost moments: benefit, positive harm,
  reference error, easy-reference mass and easy-positive harm.
- The existing bounded policy only interpolates between the fixed causal floor
  and frozen neural prediction. There is no new trajectory decoder, stochastic
  rollout, JEPA target latent, pixel reconstruction or generative execution.
- Preprocessing is fitted on critic G. Actual serialized IDs, source sites and
  normalizers are checked for all 36 heads, excluding producer F and evaluation H.
- Three actual seeds: 17, 29, 43. Fixed complete budgets; no best-seed choice or
  held threshold tuning. Exact resumed pilots and first full refits are retained.

## Data And Evaluation

- EuropeanSquares SOURCE whitelist: 149,510 overlapping windows, 74 recordings,
  12 exposed localities. Six producer/seed views per locality. The 897,060 cached
  feature rows repeat those windows, not new independent examples.
- Eight observed and twelve requested future steps, raw stride 12. Image-local
  detector/tracker midpoints in pixels, not meters, seconds or human gold.
- Future targets and validity masks are training/evaluation labels only.
  Complete12 ADE is not identified by available-label ADE or unknown-row bounds.
  Locality 082 has no final endpoint, so the all-required-locality FDE interval
  is explicitly not estimable. Older SDD/external t50/t100 scores are not rerun
  or mixed into this comparison.
- Producer F, critic G and evaluated H are disjoint four-locality groups in each
  job. SOURCE has already informed research choices; cross-fitting does not
  turn it into untouched independent confirmation. Intervals use 3,000 paired
  locality resamples after role/seed averaging.

## Findings And Limits

Primary lower effect versus Zero64 is -0.0213 pp [-0.6148, +0.6482]; known effect
is +0.4800 pp [-0.1007, +1.1025]. No reliable held utility improvement is shown.
The full-bound fitting objective improves about 11.5%, but complete risk passes
only 110/444 repeated views and worst easy-net degradation is 39.3712%.

Neighbor coverage differs between h64's original forecaster packet and the
existing partial-neighbor descriptors. The comparison cannot isolate learned
encoding from that raw coverage difference. Per-scene support is also uneven;
large window counts do not remove small-locality uncertainty.

The initial feature-cache pilot used the wrong past packet; it was corrected
and every original forecast and latent replayed. The initial head pilot requested
H instead of G, computed temporary H-derived preprocessing and stopped before
any optimizer update. That preprocessing was discarded, not reused. Both failed
attempts, initial definitions and explicit corrections are preserved; current
actual checkpoint roles pass the separate audit.

No deployment upgrade, physical-safety certificate, true 3D, foundation-model
or submission-ready claim. Stage5C and SMC are disabled. Code, configs, reports
and lightweight scalar evidence are public; raw data, caches and weights remain
private. Reconstructing public scalars is not full raw-data cold-start reproduction.
