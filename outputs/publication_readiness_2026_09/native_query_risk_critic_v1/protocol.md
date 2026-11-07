# Native Query Switch-Set Critic

At native observation-8 / prediction-12, use the same previously exposed
34 development recordings, 14 source groups, three predictor seeds, bank,
cuts, CV risk reference, and OLS4 accuracy comparator as the frozen transport
rehearsal. DroneCrowd confirmation stays unopened. Do not change the 2%,
support or undefined-risk rules.

The proposed switch sets are exactly the original joint arm, with up to two
switches per query. The old screen adds per-action estimates and checks for
a positive expected easy reference. Fit two new ridge screens to labels of
the entire proposed set: completion-lower gain, all-risk excess, easy-risk
excess, and any positive observed easy reference. The first learns from
the summed action scores alone; the second additionally sees the spread of
scores, past neighbor distances, motion summaries and relative intervention
magnitude. They use the same labels, ridge coefficient, source-balanced
weights and acceptance criterion. The operating rule requires positive
predicted gain, both excesses nonpositive and predicted support at least 0.5.
All features are available from a single query. The acceptance threshold is
fixed before training and is not a risk certificate.

For each held source, create critic-training proposals in the other 13
sources with both the outer held source and the corresponding training
source omitted from the action model. Fit the critic on those 13 sources.
Apply to the outer held source's original proposed switch sets, using no
held-source labels for its model. Freeze all decisions before the separate
readout. This is a development comparison on already exposed sources,
not independent confirmation.

Primary component readout: easy-support Brier score on identical proposals,
query critic versus recalibrated sum. Also report normalized absolute target
errors, actual local feasibility, intervention counts, the original
recording checks, and completion-lower utility versus CV and OLS4. Bootstrap
3,000 paired source groups after seed averaging. Policy arms may accept
different numbers of sets; do not interpret deployment utility as a
same-count ranking effect. A model score or positive local feasibility
does not substitute for passing recording-level evaluation.
