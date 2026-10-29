# Why This Is Not Another Global Threshold Sweep

I first checked the archived experiments before implementing this comparison.

- `european_selected_set_calibration_v1/failure_analysis.md`: iterative global
  empirical90th-percentile calibration lost utility, retained missing-outcome
  exposure, and even raised observed easy risk in one fully known reduced set.
  A smaller set is not automatically safer.
- `european_score_support_calibration_v1/failure_analysis.md`: calibrated neural
  forecasts still lost to matched protected damping. Generic support distance
  was not a reliable risk detector; two-source calibration gave no guarantee.
- `european_fixed_producer_roles_v1/conclusions.md`: matching producer supervision
  improved easy preservation but selected positive-harm risk remained uncalibrated.
- `european_rational_harm_v1/failure_analysis.md`: exact-zero Jacobians were largely
  repaired, yet the model underpredicted harm precisely in its own chosen cohort.
  Overall overprediction therefore does not solve decision-conditional error.

The new narrow contrast freezes the rational head and calibrates signed risk
excess in fixed decision cells, with source-excluded labels and a support-only
control. It tests conditional estimation beyond global offsets. It is not a
new neural architecture, dynamics advance, statistical certificate or claimed
novel calibration method. If it fails, do not keep changing cell cutoffs on the
same readout; inspect whether support, conditional bias or utility loss explains
the failure before a separately registered change.

Existing metadata gives24 source/context passports. All sources are excluded
from their corresponding upstream producer and controller.20/24 source/context
groups have only one internal-held recording; the other groups have1,10 or11.
Pooling other sources gives at least3 calibration recordings, but does not make
different localities exchangeable. Those limits are part of the hypothesis.
