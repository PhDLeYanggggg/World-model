# Predict risk support with the joint proposal held fixed

Question: does conditioning the release score on predicted availability of
an easy reference improve ranking, rather than merely changing what can be
measured? Use the existing seven-location OpenDD vehicle development bank,
seeds 17/29/43, and all 126 ordered held/calibration directions. Reuse the
frozen two-edit joint proposals and actor support heads; both weight banks
exclude the calibration and held locations. Do not change actors, paths,
amplitude, the eight observed and twelve forecast samples, or the 2% rule.

Each recording has one proposal. Compare original joint merit, support
only, and joint merit plus log predicted support. For the selected actors
with causal single-actor probabilities p and q, the support proxy is
`1 - (1-p)*(1-q)`. This is a ranking proxy, not an assumption that actor
events are independent in the real data. No parameters, weighting or
features are selected on these results. This tests release ranking, unlike
the prior OpenDD study that changed recipient selection.

Primary: for each held location, seed and calibration direction, take the
top `max(1, floor(n/4))` scores (stable frozen recording order at a tie).
Report defined risk, original passes, dual passes, and direct conservative
ADE and relative-path contrast against the original score at the same
count. Keep CV everywhere else. Average views within site before 3,000
paired location bootstrap draws with seed 12773. Compare supported merit
versus original as primary, support-only as a diagnostic. The fixed-count
readout uses the full score list, not an online release policy.

Secondary: freeze the highest-score tie-complete calibration prefix whose
proposals all pass the original three risk checks, using only the separate
calibration location. Report actual held release counts, undefined risk,
passes and paired effects with CV fallback. Counts may differ here; an
abstention is not a risk pass. No new independent external label access or
scientific release claim. OpenDD remains exposed development.
