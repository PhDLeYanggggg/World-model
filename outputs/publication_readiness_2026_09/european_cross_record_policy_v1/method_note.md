# Why Test Cross-Recording Updates?

## The Narrow Hypothesis

The previous full-gradient experiment improved training budget coverage from
35/72 to 53/72 views but reduced inner-held coverage from 4/72 to 0/72. Keeping
roughly the same fraction of held benefit and held harm suggests that the
learned rule did not separate transferable useful choices from harmful ones.
This motivates a training-objective experiment, not an inference-threshold change.

The present comparison asks whether an update that helps one training recording
also helps another. Every recording still contributes supervised targets; the
episode partition changes which recordings produce temporary weights and which
recordings assess those weights. Neither side is calibration data.

## What Is and Is Not New

Simulating domain shift within training is established prior work, notably
[MLDG, Li et al., AAAI 2018](https://homepages.inf.ed.ac.uk/thospeda/papers/li2017metaDG.pdf).
The current implementation differs: symmetric recording episodes, baseline-relative
intervention losses, per-record all/easy dual constraints, a bounded inner update,
and a matched same-record meta-update control. These choices define an experiment;
their combination is not automatically a publishable methodological contribution.

For a smooth loss, a small cross-record inner step can encourage agreement
between gradients from different recording groups. A same-record step instead
rewards improvement on its own group. Here the action is hard in the forward
pass and uses a sigmoid straight-through backward surrogate. A Taylor argument
for a smooth loss therefore does NOT prove descent of our hard policy risk.
Second-order autograd verifies the specified surrogate computation, not a
population-safety guarantee or a theorem about actual discrete interventions.

## Controlled Evidence

The decisive new contrast is cross_record_meta versus same_record_meta. Their
features, initial weights, episode sequence, row exposure, update count and
operation types match. The frozen full-gradient comparison measures the whole
meta-training change but is not compute-matched. Final weights, rather than the
best observed held checkpoint, are evaluated.

A useful result must survive full-policy safety checks as well as the utility
comparison. Fewer interventions alone cannot explain a count-matched utility
gain. Conversely, better count-matched utility cannot establish safety of the
complete deployed policy. Unknown outcomes remain bounded, not imputed as zero.

Source clustering is essential for the uncertainty diagnostic: repeated windows,
contexts and head seeds do not supply independent recordings. Bootstrap sampling
of source-averaged results describes this exposed development cohort only.
Independent evaluation remains necessary even if every development comparison
looks favorable.

## Limitations Fixed Before Readout

- Eight sources have only one fit recording, limiting within-source episode diversity.
- Episodes can include source distribution changes as well as recording changes.
- Fit-only source preprocessing is shared between episode sides; this is not
  cross-fitted preprocessing or full-pipeline OOF.
- The inherited inner-held data have already influenced past research choices.
- The forecasters and moment models remain frozen; new training concerns the
  intervention policy, not a new neural trajectory or world-dynamics predictor.
- This comparison does not add scene-joint actions, verified physical units,
  independent calibration, Stage5C or SMC.

The result section is generated only after the fixed training roster completes.
Negative findings remain part of the experiment record and do not alter the budget.
