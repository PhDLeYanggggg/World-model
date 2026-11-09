# Vehicle-only development candidate before external evaluation

The frozen neural path in the unscored Songdo bank was trained on
pedestrians. To test a better-matched candidate, use only the already
verified OpenDD v3 archives, the same seven held-location group IDs,
and three seeds (17, 29, 43). Read the archive's Car, Medium Vehicle,
Heavy Vehicle, Motorcycle and Bus tracks. Do not use separately
labelled Trailer rows as independent motorized vehicles. A query is chosen by its eight-step history
and recording time bounds, at stride 12 on the release lattice. Keep
all history-eligible vehicles in the chosen query. Queries do not overlap
within each flight, and the group split is by site, never a random
partition of adjacent windows. The future has twelve steps and is masked
when labels are missing. Do not read or score Songdo or reserved sources.

For each site, train on the other six locations. Both arms have the
same normalized 17-value history, same 64/48-unit MLP and bounded
CV-relative forecast, with a peak radius at most two observed RMS
displacements (scale lower bound 0.05 as in the prior encoder).
No other actors' futures or future-derived features enter the model.
Training samples sites uniformly, then rows within a site; weights,
batch draws, seeds and 400 AdamW updates are matched. Each 100-step
checkpoint saves weights, optimizer and RNG state.

- **Observed:** error on observed future steps, divided by the twelve
  requested steps.
- **Completion:** identical observed loss plus forecast-to-CV distance
  at every unavailable future step, with the same denominator.

Use all fixed final checkpoints, with no model or threshold chosen on
held locations. Compare CV and OLS4 on the identical full roster.
For accuracy, use paired observed gain and the reverse-triangle missing
step debit against each reference, equally weighting queries within a
location and normalizing by observed CV error mass. Report observed
ADE, available endpoint FDE and the unchanged 2% risk diagnostic; the
easy and hard cuts are the 25th and 75th percentiles of nonzero complete
CV ADE from donor sites. Use 3,000 location bootstrap draws after seed
averaging. This is development evidence and not independent confirmation.

This comparison evaluates candidate quality, not a deployed edit
selector. No new external split, threshold tuning, calibration, Stage5C,
SMC or physical safety claim. Any future use on Songdo requires the
separately chosen prospective external evaluation protocol.
