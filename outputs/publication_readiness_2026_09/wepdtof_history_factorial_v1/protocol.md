# WEPDTOF: separate the observed anchor and velocity

Use only the original 15 scored clips and original association rule.
Reconstruct the annotation observation history only for a track matched
at all eight observed positions. All four cells score the same available
future labels with the same roster and clock. For each of CV and OLS4,
cross the detector or annotation last position with the detector or
annotation velocity. Annotation inputs are diagnostic and unavailable
at deployment. The main contrast is the detector CV-minus-OLS4 ADE
gap minus the annotation CV-minus-OLS4 gap. Attribute detector-to-
annotation ADE differences to anchor and velocity by averaging both
orders of replacement. Report all factorial cells and source bootstrap
intervals. Source means weight estimable clips equally. The common
matched histories are not representative of undetected targets.

Test one detector-only control without fitting on outcomes: use the
same four-frame OLS velocity but extrapolate from its estimated
intercept rather than the last detector coordinate. Report its paired
observed ADE, FDE and p95 differences versus the frozen OLS4 on the full
online roster, including the coverage report. Also report the original
2% CV-relative recording risk test with the original easy/hard cutoffs.
No threshold selection, replacement of the existing floor, or claim
of independent validation. Previously scored outcomes may be reread
to calculate these contrasts. No other sources are opened.
