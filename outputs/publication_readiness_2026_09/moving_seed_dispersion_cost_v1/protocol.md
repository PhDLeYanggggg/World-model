# Does Seed Disagreement Help Choose When To Intervene?

The previous fixed-count pair term did not improve interaction error. This
study instead tests the marginal cost model. The question is whether between-
seed trajectory dispersion predicts harmful switches beyond a single candidate's
geometry and deviation from constant velocity. [Deep Ensembles](https://papers.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)
motivates the disagreement input; this is not calibrated ensemble uncertainty
or a claim to invent ensemble forecasting. [Regression with Multi-Expert Deferral](https://proceedings.mlr.press/v235/mao24d.html)
motivates comparing regression costs, not labeling a universally best predictor.

- Population: all 125,540 exposed A rows, 14 source groups, fixed native 8/12.
  All three EqMotion fold-2 banks have exactly these held row IDs. Each original
  4,000-update forecaster excludes the full A roster. Fold 2 is not cleared for
  DroneCrowd transfer. Keep each seed's path separately; no path averaging.
- Within each of three real seeds, leave one entire A source out when fitting
  marginal costs. Balance fitting sources, then queries, then rows. Train only
  on available-label costs; do not impute unknown loss. Original held labels
  never enter their cost fit. Other A sources are exposed development, not
  independent confirmation.
- Same two histogram-boosted regressors per arm: expected signed ADE gain and
  expected positive harm, each divided in fitting by the past-only scale
  `max(1 px, mean CV rollout distance from origin)`. Restore original pixel
  costs for decisions. Both arms use observed speed/acceleration/path/crowd and
  their same candidate-vs-CV geometry. Treatment adds four dispersion features
  from the fixed three-seed paths; all budgets and paths are unchanged.
- Score gain minus positive harm, with no tuning. Both arms switch exactly
  `min(2, n_agents-1)` per query, with row-ID tie breaking. Freeze all three
  seeds' actions and costs before evaluation. No training of new forecasters.
- Evaluate with the prior A partial-label paired statistics and frozen easy/hard
  cuts. Average paired changes over seeds within each source, then bootstrap
  sources 3,000 times with seed 73107. Report all source and recording views,
  raw/fallback/selected ADE gain, conditional known-label cost error, harm,
  reference support and equal count. Full 12-step completion-lower utility
  difference and known easy positive-harm difference are the two component
  endpoints. Progress only if the lower endpoint of the utility interval is
  positive and the upper endpoint of the easy-harm-increase interval is <=0.
  Any undefined source endpoint stays undefined; no defined-only primary.
- This is a model-side diagnostic. The fixed count is not a calibrated risk
  budget. No full deployment pass, no changed 2% all/easy positive harm or
  signed easy-net and support requirements, no identity certification, no new
  calibration or confirmation labels. No SMC or Stage5C execution.
