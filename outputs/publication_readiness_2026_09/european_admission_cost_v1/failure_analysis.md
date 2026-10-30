# Failure analysis: online admission loss

## What was actually tested

One controlled extra loss on the model's current supervised winning action,
not another threshold sweep or larger neural dynamics model. All 54 heads ran
the full 2,000-update budget. The 36 original/upper controls reproduce full
scientific states exactly. The new head has an exact complete refit; all 72
policy groups have separate exact inference replay. This rules out a changed
sampler, preprocessing, proposal bank or shorter optimization budget as the
source of these contrasts. It does not establish independent generalization.

## Primary repair fails

Against upper score, the known ADE increment is -0.075404pp with locality
bootstrap CI[-0.245661,+0.026133]. The completion-lower increment is
-0.120265pp, CI[-0.290321,-0.012139]. Seed increments are +0.063829,
-0.121809 and -0.168232pp. Locality110 accounts for the largest negative
change (-0.951312pp); six localities improve, five decline and one is zero.
These are fitting-side post-training diagnostics, not new model-selection
rules. No locality or seed is discarded.

Recording support moves41/444 to46/444. Known easy violations move312 to308,
known all violations156 to149, while42undefined easy views remain. This is
not a safe policy. Selected known views move262370 to262035; unknown
selections3658 to3636, but their envelope mass increases62829.450789 to
63744.494375. Fewer unknown selections do not imply lower unknown risk.

## The targeted mechanism remains

Easy false admissions fall only33241 to33143 (98 fewer overlapping views).
Numerator-only witnesses fall32532 to32446. Observed selected easy harm falls
19341.876351 to19255.862315, but predicted easy harm also falls255.500950 to
254.215456. The magnitude gap remains over75-fold. The extra loss did not
solve the selected numerator failure. Predicted easy-reference mass on known
noneasy selections remains60078.085164. All-channel reference inflation also
persists:4135728.653751 predicted versus2299001.517273 observed.

This does not prove that the extra loss never activated, that its unit weight
is too small, or that longer training would succeed. Separate per-step
activation/auxiliary-gradient traces were not saved; total loss alone cannot
resolve these hypotheses. Label-informed arithmetic witnesses identify how
fixed actions were misadmitted, not causal effects of an intervention.

## Positive secondary evidence must stay secondary

The same admitted-score head benefits from allowing its frozen neural action:
known increment+0.194407pp, CI[+0.049385,+0.386839]; completion-lower increment
+0.164782pp, CI[+0.019557,+0.359793]. All three seed means are positive for
this contrast. It is a same-head action-availability diagnostic on exposed
fitting data, not a new forecaster training effect, independent confirmation
or complete-policy safety. The preceding upper arm already had a positive
secondary contrast. The new loss does not receive credit for creating it.

Hard pooled gain rises only1.330585% to1.343824%, below the original1.560986%.
Easy net gain remains about4.896%, but net benefit cannot cancel positive harm
or hide308violating recording views. The full-roster FDE remains not estimable.

## Consequence

Do not deploy this repair, promote an independent role, relax2%, discard
unknown rows or choose a winning test slice. The next discriminating check is
the activation and channel-gradient contribution of the extra loss on fixed
past-only fitting batches, alongside the selected tail and reference support.
Use that to distinguish ineffective optimization pressure from inadequate
conditional features or a poor proposal/error distribution before choosing a
different repair. A new mechanism needs a new fixed registration. The current
result cannot establish calibrated safety, world dynamics, true3D, metric
prediction, seconds-level horizons or a foundation model. No Stage5C or SMC.
