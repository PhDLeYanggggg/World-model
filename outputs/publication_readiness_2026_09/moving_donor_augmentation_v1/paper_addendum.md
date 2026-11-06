# Source-Held Training Donors Improve Neural Risk Information

The previous fixed-budget continuation improved fitting but did not improve
source-held risk decisions. We therefore test a data-side repair. Six randomly
initialized EqMotion crossfits within the existing producer-training partitions
provide source-held forecasts; their teacher training groups exclude their
own donor sources and each parent's outer calibration/evaluation groups.

We form six fixed interpolation candidates per donor history, retaining all
causally nonidentical candidates. Original120 motion, rollout/history and
neighbor features are constructed before future-label filtering. Future labels
are supervision only. There are736,296known candidate records and16,944unknown
legal candidates, with the latter not misclassified as safe. The experiment
does not use a warm router exposed to the query source to label these donors.

In eight already-exposed own-B source-held folds, the same9,890-parameter MLP
is trained with own-only, A-only or50/50donor mixtures. Each receives4,096updates
and source/query-balanced sampling. All24primary fits and24complete exact refits
finish before the final development readout. All methods are evaluated with
the same original fitting-only normalization and zero decision threshold.

The mixture lowers source-macro decision cost20.71%,32.27%,10.69% relative to
the matched own-only models, and lowers risk MSE64.34%,16.73%,12.00%. All eight
source folds improve total decision cost. These component results support the
practical donor treatment, not a pure causal claim about source count, since
candidate distribution and crossfit teacher training size also differ.

The result does not establish safe deployment: relative to the conservative
forest, total decision-cost changes are+4.96%,-10.86%,+2.24%, while false-safe
cost increases in every context. Source060 also has more harmful admissions
than matched own-only despite lower total cost. We preserve these negative
comparisons and do not tune an admission threshold on them. A lower regression
error is not a complete risk certificate.

This is a one-seed, exposed-development information result, without new
independent confidence intervals or a three-seed forecasting experiment.
The next step is seed replication and fitting-source uncertainty estimation,
not claiming a deployable model. The original2% full-policy contract, empty
reference NA and original24-group main NA are unchanged. Native image-coordinate
8/12silver annotations do not establish meters, verified elapsed seconds,
true3D or a foundation world model. Stage5C and SMC remain off. The compiled
main paper is unchanged and is not submission-ready.
