# Existing Evidence And Why This Is Not Another Calibration Sweep

- [Matched cost forest](../european_multiaction_forest_v1/readout.md): all 54
  current seven-action fits complete. Mixed held utility is effectively zero,
  with only 182 neural choices versus 31,959 for its neural-only expert. The
  next check uses fitting packets only; it does not retune that readout.
- [Fixed-region fitting diagnosis](../european_multiaction_forest_v1/fitting_cost_summary.json):
  mixed forest/neural EH ratios in identical frozen neural selections are
  4.576410/0.021367 despite near-unit all-legal medians. This is a measured
  operating-point difference, not uniquely proved architecture causality.
- [Earlier selected-set recalibration](../european_selected_set_calibration_v1/conclusions.md):
  source-OOF complete support remains 23/72 and conservative utility changes
  -0.00678063%. Fewer violations on a smaller population were not a repair.
  Historical report bytes are reused, not a fresh rerun of the old CREATE job.
- [Other analogue and conditional controls](../european_selection_conditioned_analogue_v1/prior_controls.md):
  matching or conditional risk estimates may overshoot, lose coverage or fail
  after finer selection. No such estimator, cell or cutoff is retrained here.

The present fixed restriction isolates physical-action competition while
holding mixed predictions identical. Separate frozen-expert composition then
isolates the operational effect of using bank-specific costs, with its extra
capacity/compute stated. It uses no oracle label for decisions, new donor
population, global scalar, monotone margin or held-tuned threshold. A negative
result remains negative and no independent world-model claim follows.
