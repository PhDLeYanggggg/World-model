# Learn When Not to Edit, Without Leaking the Held Source

The previous causal joint head could not make a forced eight-edit quota safe.
This is a different question: does an explicit prediction of support and
positive harm improve optional selection, relative to a gain-only ranker at
the same query count? The recording risk definition is not changed.

- Rotate the existing three fitting-source folds through training,
  calibration, and held comparison (three source groups each, nine total).
  No A labels. Train and calibrate on analytic CV-relative proposals only;
  any archived predictor for the other roles may have trained on the held
  sources. The held bank uses its original six-source-excluded PCT/EqMotion
  plus the five analytic candidates. Scores use no candidate family ID.
- Each direction may use fixed CV blends of .05, .15, .35 and 1. Sample
  these prospectively during fitting. Nominate a single action per row by
  expected average paired gain before screening; use the same nominees
  for independent, joint, and exact-count gain-only comparisons.
- The factorized head predicts normalized paired utility, observed CV
  reference, known positive harm, easy shares of both, and mutually
  exclusive probabilities of positive easy support or wholly missing
  future. The missing debit uses predicted probability times causal peak
  displacement. These are estimates, not guaranteed bounds.
- Fit a shared edge model on sampled analytic pairs. Optional independent
  and coupled arms both screen the same rows with predicted all/easy risk
  excess and easy-support probability, and choose exactly the same number
  up to eight per query. Gain-only picks the same number from all legal
  nominees; a fixed forced gain arm gives the previous intervention scale.
- Report the fixed prediction screen (`tolerance=0`, `p_easy>=.10`), a
  fixed relaxed diagnostic (`tolerance=.03`, `p_easy>=.10`), and a
  separate calibration-selected screen. Before any held scoring, search
  only the fixed tolerances `[-.005, 0, .002, .01, .03]` and minimum easy
  probabilities `[0, .10, .25]` on the calibration recordings. A setting
  qualifies for the screened independent arm only with nonzero edits, all recording risk passes and positive
  source-macro conservative utility against CV and OLS4. A failed search
  keeps exact CV. Empty support remains undefined; do not credit it as pass.
- Three seeds, fixed 1,000 updates per row/edge head, batch 128, checkpoint
  every 100 updates, native CPU with zero workers. Freeze all held decisions
  before held scoring; prediction comparisons use source-group bootstrap
  after seed averaging. Do not evaluate A or independent confirmation in
  this study. No deployment, Stage5C, SMC, or time/metric claim.
