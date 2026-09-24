# Asset Inventory Correction, 2026-09-24

The previous turn incorrectly described the pair-excluded full-EqMotion bank as
missing. It already exists and was completed on 2026-09-22:

- `eqmotion_nested_v1`: 18 pair-excluded predictors, 12 aligned cost views;
  analysis SHA256 `8367e7bd5628e01fa04d1c5451ec2c8816ddbee60dac07b1cd36a16b0eb05f77`.
- `eqmotion_cost_refit_v1`: 36 predictor-specific heads, including the twelve
  full-forecast bounded-fraction heads needed for the present comparison;
  analysis SHA256 `260700b79a0e0bda415fc54b4e09df66ca3c2b4c2732573ec5c32d2d7985dd9d`.
- The current `run_m3w_eqmotion_cost_refit.py --audit-only` succeeds, validating
  841 dependency bindings and all twelve cost views. Prior replay/arithmetic
  records are hash-bound. This is cached verification, not new predictor training.

The omission in the completed protected-motion experiment remains real: EqMotion
was not among that experiment's seven action families. What remains to be added
is a common-protocol comparison and twelve sampler-matched **full-EqMotion forest
heads**, not an eight-hour repeat of the existing predictor bank. Earlier forests
on ramp/uniform candidates do not supply this full-forecast comparator.

The frozen registration is retained unchanged to preserve experiment hashes.
Its statement that the producer bank was missing is superseded by this correction.
The training, choices and numerical results of the protected-motion experiment
are unaffected. No historical result is relabelled fresh or independent.
