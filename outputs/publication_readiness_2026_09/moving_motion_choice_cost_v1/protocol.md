# Cost vs. Order at Identical Motion Edits

The previous within-history backtest ranked many pairs correctly at lead 12
but its winner lost to OLS4. I test whether the magnitude and temporal location
of signed errors are useful supervision, without changing which forecasts are
edited, their peak displacement, or the inputs seen by the classifier.

- Anchor: unedited OLS4. Candidate directions: CV, OLS8, damped OLS4,
  stabilized turn, and stationary. Retain only actions that can be shrunk to
  exactly half the RMS observed one-step displacement from OLS4; never upscale.
  Force one edit only on rows with at least two legal actions. All arms modify
  those exact same rows. This is a conditional shape-choice experiment, not a
  deployment policy or an intervention-rate search.
- Three identical MLPs: winner cross-entropy (split exact ties), whole-path
  expected gain and harm, and expected gain and harm for leads 1-4, 5-8, 9-12.
  Each cost is in units of its own causal peak edit. Gain and harm are split
  per observed lead before aggregation. MSE on both targets uses observed-lead
  count weights; the two cost arms choose with equal weight over 12 leads.
  Rank chooses by its logits. No arm sees the future or future mask at inference.
- Fit nine previously approved source groups only, three seeds, 1,000 AdamW
  updates per seed and arm, identical initial states and site-balanced samples.
  Fixed terminal checkpoint. Freeze all A action arrays before reading outcomes.
- Primary: lead cost minus pooled cost, source-macro paired full-grid lower
  gain in pp of CV error mass. Secondary: pooled versus rank, lead versus rank,
  comparisons to OLS4, lead 12 alone, and observed-only menu regret. Bootstrap
  3,000 times over 14 source groups after averaging seed results. Missing leads
  debit their paired prediction distance; endpoint FDE is not inferred when
  the endpoint is missing. No threshold search on readout A.
- Original CV-referenced 2% recording rule and undefined-support handling
  unchanged. This is reused development evidence (34 recordings), not a new
  independent confirmation. No DroneCrowd confirmation access, deployment,
  latent rollout, SMC, metric/seconds claim, or submission readiness change.

Predicting useful costs rather than labels is an established direction
([Elmachtoub and Grigas, 2022](https://doi.org/10.1287/mnsc.2020.3922));
this experiment does not implement SPO+ or claim that idea as novel.
