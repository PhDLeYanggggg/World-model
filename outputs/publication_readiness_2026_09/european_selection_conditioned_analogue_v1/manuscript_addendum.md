# Exploratory Note: Matching the Donor Population Does Not Calibrate Harm

We examined whether pooled analogue risk estimates were distorted by a mismatch
between an already selected query cohort and all eligible reference labels.
Supervised donors were restricted to samples selected by the same frozen causal
policy. Query selection, feature normalization, recording exclusion and the
32-neighbor rule remained fixed. All five inherited prediction controls were
reproduced exactly, and all72 fitting views retained full reference support.

Compared with full-bank nearest neighbors, policy-conditioned donors reduced
easy signed-risk MSE from0.004741 to0.002366 and all-risk MSE from0.297407 to
0.101728. However, easy MSE remained7.65 times the original head's0.000309;
none of18 context/head-seed pairs beat both required comparators. The conditioned
nearest estimator was worse than its donor-mean control and only0.78% better
than its shuffled-label control on easy MSE. The registered magnitude screen
failed and no new policy was deployed.

Event/magnitude accounting reveals why aggregate probability is insufficient.
Mean predicted easy-harm event probability was6.2746%, close to the observed
6.1335%, while total predicted easy harm remained3.88 times observed. Harm error
fell on realized non-events but rose on realized positive events. These are
descriptive realized-outcome decompositions, not a claim that non-events should
receive zero probabilistic risk. They show that matching prevalence alone does
not resolve conditional allocation and magnitude errors under this construction.

The study uses exposed-TRAIN recordings with upstream model/selection exposure,
so reference exclusion is not whole-pipeline out-of-fold evidence. There is no
new held or independent evaluation, confirmatory CI, neural training or utility
claim. The protocol remains obs8/pred12 at rawstride12, detector-silver pixel/local
coordinates, and2% positive selected harm/reference. No metric, seconds, human
gold, physical safety, true3D or foundation interpretation is supported. Stage5C
and SMC remain disabled. This note is not a completed submission-ready paper.
