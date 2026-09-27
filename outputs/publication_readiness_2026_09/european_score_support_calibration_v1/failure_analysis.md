# Calibration Protects Easy Cases but Does Not Recover a Neural Advantage

## What Was Actually Changed

This experiment fits no new networks. It reruns the nine frozen forecasting
banks and matched cost heads, equally ensembles four three-source risk heads,
restores movement/utility/easy guards, and calibrates a fixed threshold grid on
two completely fitting-excluded sources. The other two sources supply readout
only for that view. All six rotations, both objectives and three seeds remain.
Scores and actions were committed before their respective outcome use.

This is not a rerun of the old risk-only screen: static guards are now present,
heads are fixed ensembles and the evaluation population is the outer roster.
Therefore improvements over the previous turn cannot be attributed solely to
calibration. The within-experiment factorial comparisons isolate its effect.

## Primary Failure

Signed calibrated-supported neural versus equally protected damping ADE gain is
**-0.6699% [-0.9742%, -0.3964%]**. All three seed intervals favor damping; eleven
of twelve locality point estimates favor damping. This does not support the
requested neural contribution, despite both policies improving over CV.

Neural all-ADE gain vs CV is0.1666% [0.1206,0.2205], hard gain0.0478%, coverage
9.7901%. Matched damping improves0.8259% [0.5478,1.1164], hard0.4423%, coverage
35.6017%. A positive comparison against CV alone would hide the failed stronger
control. All nine prespecified neural/damping policy contrasts have negative
intervals. The old full policy is reproduced, not replaced or recertified.

## What Improved

The complete static-plus-calibrated rule preserves easy within2% in every
observed neural held view, and harms no reference-exact cases. Mean easy gain
is3.4125%; the worst view gain is0%, including fallback views. Signed neural
guarded risk violations fall from75/216 to9/216 with calibration+support.
These counts are dependent calibration/source/seed views, not216 independent
scenes. Calibration-only has6 violations; support does not uniformly improve it.

The new controls thus demonstrate an empirical safety/coverage tradeoff, not
zero interventions masquerading as model improvement: coverage remains9.79%,
but45/108 calibration groups choose complete fallback. The other63 groups
mostly keep cutoff0; only3 supported groups choose-.0005. Damping has no
all-fallback groups and no observed risk violations with the complete guards.

## Which Factors Failed

- **Calibration suppresses gain:** signed calibration-only versus guarded ADE
  change is -0.1034% [-0.1511,-0.0562]. With support fixed, the change is
  -0.0663% [-0.1064,-0.0306]. This is a price for stricter observed safety,
  not a demonstrated accuracy improvement.
- **Generic support distance is not a risk detector:** adding the fitting99%
  RMS-z gate reduces neural guarded ADE by0.0500% and changes coverage by only
  -0.1247percentage points. Adding it after calibration reduces ADE by0.0130%
  and leaves more risk-violating views (9 vs6). The pointwise subset is smaller,
  but its harm/reference ratio need not improve.
- **Selected-source extrapolation remains:** retained neural risk violations
  occur in locality082/seed17 and locality020/seeds29,43, repeated across three
  calibration pairs. Ratios are2.0271%,3.4295%,2.6437%, respectively. All still
  have positive net ADE and easy gain. Positive-harm budget is not net error.
- **Damping already satisfies the calibration screen:** every damping cutoff
  remains0. Its calibrated and uncalibrated actions are identical; there is no
  evidence that calibration caused its benefit. Support slightly removes useful
  damping actions as well.
- **Small source count limits guarantees:** only two calibration localities
  constrain each view. Do not use overlapping windows as independent calibration
  units, claim conformal coverage, or increase the budget to force a pass.

See [paired factor contrasts](factor_diagnosis.md) and
[all cutoff/fallback counts](calibration_summary.md). All actions remain frozen.

## Next Evidence-Producing Step

Stop this cutoff/support sweep. The fixed score families now have evidence of
limited safe neural usefulness, while the damping controller is stronger.
Next audit the **incremental** neural opportunity over protected damping,
rather than again learning benefit relative only to CV. Use diagnostic future
labels solely to quantify attainable gain, harm and where current causal
motion/neighbor features distinguish them; never use those labels as inputs.

If incremental headroom exists in fitting-excluded views, register a paired
gain/harm target against the stronger floor with the whole producer chain
excluded. Do not train on in-sample learned-floor outputs and call them OOF.
If incremental opportunities are not identifiable from past geometry, prioritize
observation/scene context or label quality rather than another threshold sweep.
This is the next test, not a completed new model or a promise it will work.

No new independent evaluation was opened. Silver image-local obs8/pred12 at
raw-frame stride12; no metric, seconds, physical safety, true3D or foundation
claim. No deployment change, Stage5C execution or SMC.
