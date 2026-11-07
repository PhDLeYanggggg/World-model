# Joint Path Assignment Before Collapsing The Shortlist

Use the frozen witness shortlist, the same path bank and risk-cost
coefficients, and the original per-query switch count. Enumerate all
legal one- or two-recipient assignments without first choosing one
path per person. A path cannot be used twice on the same recipient.
Calculate predicted CV gain and OLS4-anchored graph energy with the
same normalization and graph weight as the single-path decoder.

Primary contrast: expanded joint choice vs. original single-path
choice on exactly the original gated active queries and counts, using
only assignments accepted by the unchanged predicted-risk gate.
Compare a matched additive arm with the same graph marginals and no
pair cross term. Also compare the ungated joint search at the raw
matched counts, and separately the screen-first joint strategy which
may activate previously rejected queries. The latter is not an
equal-intervention comparison and may still violate recording risk.
There is no new learning, recalibration, or use of later queries.

Freeze all assignments before opening scoring labels. Replay the
old risk summaries exactly. Score the original CV/OLS4 utility,
all/easy positive-harm and easy support, actual intervention count,
and the relative-trajectory ADE/FDE contrasts with missing-pair bounds.
Use 3,000 paired source bootstrap draws, after averaging three frozen
forecaster seeds across the same 34 exposed recordings and 14 sources.
The task remains eight observed / twelve predicted annotation steps
in image pixels. No external data access, new deployment, Stage5C or SMC.
