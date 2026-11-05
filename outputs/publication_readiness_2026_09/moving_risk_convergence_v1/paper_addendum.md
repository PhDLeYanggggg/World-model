# Development Addendum: Separating Fitting From Source Transfer

We evaluated the hypothesis that insufficient optimization explained the previous
neural risk model's failure. Eight saved risk heads were continued with architecture,
optimizer settings, causal features, fitting-only preprocessing, reference IDs and
decision rules unchanged. Each received at least 4,096 additional optimizer updates,
rounded to complete epochs, totaling 32,871 updates. All continuations were completed
and reproduced before evaluating the single final checkpoint. No held outcomes were
used to select an epoch or threshold.

Fitting loss decreased in all eight folds, whereas source-held plain risk MSE improved
in only two. For example, source076 fitting loss decreased from 0.525897 to 0.000365,
while source-held error changed from 7.738072 to 7.787639. Continued risk transport
produced decision-cost reductions of -4.719181%, -68.647572% and -6.692049% relative
to the verified forest control. The fixed component screen passed zero of three
contexts; this is not an improvement in trajectory ADE/FDE or full-policy safety.

The result weakens an undertraining-only explanation for this model. It is consistent
with overfitting and source-conditional mismatch, while not identifying a unique cause.
Verified lineage shows each fitting bank includes only one or two source components;
overlapping windows cannot substitute for independent source diversity. Small in-sample
donor residuals also leave little useful correction when query predictions extrapolate
poorly. Cross-fitted residuals and expanded compatible source banks are future questions,
not completed repairs.

No fold met the registered operational fitting-plateau diagnostic, so convergence is
not claimed. This remains one-seed, exposed own-B development with silver supervision
and no new independent test. Exact refits establish computation, not independent evidence.
Original risk and support rules remain; empty support is NA and no identity exemption
is introduced. Deployment is unchanged and the original 24-group primary is not passed.
No neural trajectory, metric/seconds-level, true3D, foundation or submission-ready claim
follows. This addendum records a negative optimization diagnostic; the compiled main
paper is unchanged. Stage5C and SMC remain disabled.
