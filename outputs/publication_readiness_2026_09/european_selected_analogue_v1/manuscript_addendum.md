# Exploratory Manuscript Note: Information Is Not Calibrated Risk

We examined whether frozen causal features carry harm information within an
already selected fitting cohort. For each query recording, we standardized
features using eligible rows from other recordings of the same fitting source
and retrieved32 nearest labeled donors. Controls used a donor mean or identical
neighbors with paired harm labels permuted within donor recordings. The original
selection and query roster, including unknown outcomes, were unchanged.

The study comprised72 context/source/head-seed views, of which only24 had full
donor support. In the supported-view diagnostic, nearest-neighbor easy signed-risk
MSE was0.003420, compared with0.008225 for donor means and0.013426 for permuted
labels. These differences suggest local information in feature matching, but
are not evidence of a calibrated risk model: nearest-neighbor MSE exceeded the
original risk head's0.000876, and aggregate predicted easy harm was5.48 times
observed. Complete-roster metrics remained undefined because48 views had no
alternative fitting recording. The preregistered information screen did not pass.

These are exposed-TRAIN diagnostics, with repeated cohort occurrences, no new
policy evaluation and no confirmatory confidence interval. Upstream prediction
and selection exposure precludes a whole-pipeline out-of-fold interpretation.
The observation motivates distinguishing feature information, magnitude learning
and reference support; it cannot support a headline claim of safe world modeling.

Protocol: obs8/pred12, rawstride12, detector-silver, image/local coordinates.
No metric, seconds-level, human-gold or physical-safety interpretation. Stage5C
and SMC are not executed. This note is not a completed paper or submission claim.
