# Frozen OpenDD-to-TUMDOT-ING Transfer Diagnostic

Does the direct self-history forecast retain any incremental utility on a
different aerial tracking release, at the same causal observation length and
the same blend amplitude as a simpler motion model?

**Scope.** One diagnostic sample: lexicographically first raw CSV per
intersection and day, nine files, no continuous timelines or substitutions.
All belong to one three-day Ingolstadt campaign. These are not nine
independent sites or an independent confirmation cohort. No source is assigned
a new formal role. No DroneCrowd confirmation data are accessed. Its approved
15/30 roles and unchanged 2% rule still apply.

**Roster.** Greedy earliest nonoverlapping intervals containing at least two
pedestrians with eight exact past samples, using stride 12 on the verified
29.97 Hz release grid. Forecast all history-eligible pedestrians, including
those with no observed future. The twelve future samples are for scoring only.
Use the published projected axes after translation; no released velocities,
goals, images, traffic peers, or future positions are model inputs.

**Arms.** Retain the 21 hash-bound OpenDD self-history checkpoints. For each
existing seed, average seven OpenDD producers' paths equally before blending.
No new fitting, threshold selection, or normalization on TUMDOT-ING. Compare
CV, OLS4, the direct forecast, and the 5% and 25% neural blends with CV. The
primary coefficient is 5%, the modal nested coefficient from OpenDD; the other
two are descriptive. Compare against equally blended past-only hindcast
choices from OLS4, OLS3, OLS8, damped4 and bounded acceleration4 at the exact
same recipients. Eligibility is based only on nonidentity disagreements.

**Readout.** Report paired missing-step-debited gain versus CV, OLS4 and the
matched control with observed CV error mass as denominator; retain available
ADE/FDE as descriptive. Report each file and seed, and average the three seed
views within file before equal-file and per-day summaries. Do not present the
three days from one campaign as a population confidence interval. Reuse the
original recording-level 2% algebra with easy and hard cuts from pooled
positive complete OpenDD CV ADE (quartiles), not target outcomes. Undefined
risk remains a failure. The legacy scorer's unit-dependent output labels are
not used. No deployment, Stage5C, SMC, or new SDD metric/time claim.
