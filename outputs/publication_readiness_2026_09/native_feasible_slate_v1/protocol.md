# Direct Feasibility Ranking On The Fixed Native Slate

Keep the 34 exposed development recordings, 14 source groups, 1,613 queries,
and three actual forecaster seeds. Use the previous native observation-8 /
prediction-12 forecast bank, fitting-only cuts, shortlist of up to eight
replacements, constant-velocity risk reference and OLS4 accuracy comparator.
The recording-level 2% rule and support/NA handling are unchanged. No reserved
external input or outcome may be used.

Train one fixed gradient-boosted classifier per seed and held source to predict
the joint event of positive completion-lower gain, nonpositive all/easy risk
excess and positive observed easy reference for a candidate set. All features
are current-query motion summaries, baseline forecasts and their disagreements,
and fitted additive estimates; the event uses labels only during training or
scoring. Donor slates are constructed with both the scoring source and the
donor source excluded from the additive fit. Training weights balance sources,
queries within sources, and options within a query. The classifier settings
are fixed before fitting, without held-source tuning.

Primary readout: source-paired top-one event hit rate across all queries,
versus the existing support ranker on exactly the same slate and agent counts.
For the operating-policy control, preserve the exact active-query mask of
the old gated support policy, and let the new ranker choose only proposals
that pass the same frozen predicted gate. This isolates ranking without
changing when or how many agents switch. Report paired utility, relative
path ADE and completion bounds, the original recording check, and the
change in choices. Use 3,000 paired source bootstrap draws after seed
averaging. A local event hit does not certify a whole recording.

Source-excluded development evidence cannot promote deployment, establish
physical safety, or substitute for independent validation.
