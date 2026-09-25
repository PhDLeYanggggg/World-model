# Failure Analysis

## Supported Findings

1. Complete producer exclusion is now executable. Calibration and outer sources
   never enter their prediction/cost-producing fitting chain. This repairs the
   experimental design, not the neural prediction quality.
2. Empirical calibration partly transports and partly fails. The neural grid's
   positive-harm restrictions pass all36 calibration assignments but only27
   outer assignments. A rule that fits four localities is not a per-locality
   guarantee. Keeping positive harm separate from net gain exposes the failure.
3. Strong motion remains the more useful controlled candidate. All36 neural
   contrasts against equally protected damping have negative ADE point estimates;
   none has a positive interval. Hard and easy contrasts have the same sign.
4. Preservation is not uniform. Neural observed safety passes2/12 without
   calibration and5/12 under each calibration method. Some net easy violations
   fall, but rare zero-CV cases are still changed incorrectly.
5. Calibration generally removes benefit with harm. Population rescaling has
   eight strictly negative neural gain-change intervals; the selected grid has
   six. Neither calibration method has a positive neural gain-change interval.
6. The raw candidate is itself variable: neural ADE gains2.4167--3.0561% overCV
   have wide intervals crossing zero, whereas raw damping's3.9755% interval is
   positive. Scoring is not the only possible bottleneck.

## Hypotheses Not Yet Isolated

- Inner-to-final producer shift: heads learn from two-locality OOF predictors
  and deploy with a four-locality predictor. This design prevents exposure, but
  conditional cost distributions need not match. Calibration alone does not
  identify this mechanism separately from scene shift.
- Insufficient causal discriminability: the available motion/neighbour history
  may not distinguish future neural errors well enough. Hindsight oracle gain
  is not proof that the gain is predictable from past inputs.
- Tracking artifacts and incomplete future support: these are detector tracks,
  not human-gold or verified causal online observations.7,047 windows have no
  supported future label, and only193,705 have complete futures. Future support
  cannot be used as an inference-time filter to make the scores look safer.
- Small calibration support: four localities per assignment and very rare
  zero-CV events limit what can be learned or certified. Overlapping windows
  must not be counted as independent calibration samples.

## Next Falsifiable Step

Before another fit, measure matched source-locality cost/score errors for the
inner producers versus frozen final producers, retaining all seeds and the
existing source roles. Separate population bias, bias among selected rows,
oracle opportunity captured, and added harm. This is an attribution study,
not permission to refit the frozen thresholds on outer errors.

Only then register one supported change, such as producer-consistent cost
training or uncertainty-aware abstention, with a matched damping control.
Increasing model size, changing the2% tolerance, opening reserved outcomes or
calling empirical threshold fitting conformal would not resolve this result.

Source-development image-pixel obs8/pred12 rawstride12 only. No deployment,
independent certification, metric/seconds, human-gold, true3D, foundation,
Stage5C or SMC claim. The experiment establishes neither a joint-control
contribution nor CVPR submission readiness.
