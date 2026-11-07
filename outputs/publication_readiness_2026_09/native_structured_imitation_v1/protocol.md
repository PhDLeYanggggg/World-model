# Distilling Exact Recording Decisions Into Past-Only Choice Scores

On the same exposed Moving A recordings and 8/12 annotation-step
pixel task, construct training targets from the verified exact-count
CV-utility optimizer over the full native bank. Keep the original
2% all/easy harm, signed easy and positive-reference conditions.
Only original-scorer-verified feasible teachers contribute labels.
These teachers use future outcomes and are never inference policies.

Each teacher is a function of one recording, frozen native paths,
the training-derived risk cuts and the original query counts. It
does not use a learned cross-recording recipient nomination. This
allows the outer source-excluded student fit to use other sources'
teacher labels without a held-source dependency through the teacher.
No label-aware action arrays are published.

For each selected action, train two pairwise comparisons: against
legal actions of unselected recipients, and against alternative
legal paths for that same recipient. Regress the past-descriptor
difference to one with ridge penalty 0.01, balancing the two
comparison types when both exist, queries within sources and
sources within the fold. Add frozen path-family indicators.
For each seed and outer source fit only on other sources.

Compare the original gain, support-first and risk-shortfall
choices to two uses of the same student: selecting recipients
with fixed original nominations, or selecting both recipients
and their paths. Enforce the same genuine intervention count per
query. Also apply the original predicted gate separately to the
student's full-bank choices; screened rates are not matched.
Freeze all inference actions before scoring. Main contrast:
full-bank student versus gain, with original recording passes,
conservative CV/OLS4 utility, relative-path errors with missing
pair-step bounds, and 3,000 paired source draws after
seed averaging. The study is exposed development, not calibration
or independent confirmation. No threshold tuning, new external
access, deployment, Stage5C or SMC.
