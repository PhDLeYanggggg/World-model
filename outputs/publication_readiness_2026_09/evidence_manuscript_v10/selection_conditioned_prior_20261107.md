# Selection-Conditioned Risk: Related-Work Note

[Wang et al., ICML 2026](https://proceedings.mlr.press/v306/wang26a.html)
express an error probability conditional on selection as a linear
expectation constraint. Their single-model Theorem 3.1 uses exchangeable
calibration and test examples and gives marginal error control conditional
on acceptance, with the zero-acceptance case vacuous. They also study
two-model routing. I checked the formulation and theorem statement in
the first six pages of the official paper, not a third-party summary.

Our algebraic comparison of harm with 2% of reference error is not a
novel risk-control principle. The different empirical question is whether
past-only forecasts permit a useful *joint multi-agent intervention*
when the original benchmark requires nonzero continuous reference mass,
two harm ratios within each recording, and conservative handling of
missing future labels. Neither their marginal theorem nor their QA/VQA
experiments imply any of those recording-level checks. The fixed-slate
ranker estimates a supervised local event and is not calibrated by their
procedure; no finite-sample guarantee is claimed.

[Mao et al., ICML 2024](https://proceedings.mlr.press/v235/mao24d.html)
also establish multi-expert deferral for regression. Deferral, linearizing
a ratio, or adding a boosted ranker cannot be positioned as standalone
novelty. A submission needs defensible empirical evidence for useful
joint choices at matched counts, retained reference support, and separate
source confirmation. This note changes no endpoint, split or threshold.
