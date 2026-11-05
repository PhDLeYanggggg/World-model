# Complete-Fitting Optimization Diagnostic

We tested whether insufficient fitting of the neural intervention-risk head
could explain the failure of previous training-schedule changes. Three
source-separated continuations used the same 492 causal inputs, 32,130-parameter
network, labels and empirical query weights. Full-fitting LBFGS with double
precision and no weight decay replaced stochastic optimization. This bundled
change is an optimization diagnostic, not an isolated optimizer ablation.

Each model completed200 parameter-changing outer calls. Mean training MSE
fell31.98%,37.90%,41.13%, respectively. However, cost-weighted wrong-sign risk
increased in at least one channel in every context: all-channel cost+8.32%
in the first, easy-channel cost+2.37% in the second, and both+2.57%/+6.34%
in the third. Thus improved regression fit did not establish more useful
intervention decisions, even on the fitting data.

Attribution showed heterogeneous failures: additional false rejections in the
first context and added false-safe cost in the others. Conservative targets
for missing outcomes contributed40.19%,49.92%,75.14% of the MSE decrease;
observed-outcome MSE also decreased. This is a descriptive decomposition, not
proof of a causal defect in bound supervision. The bounds were not removed.

All contexts failed the prospective fitting screen. Consequently no new held
prediction or threshold search was performed. Earlier held failures remain
unchanged. Exact resume, a complete independent refit and fitting attribution
replay establish computational reproducibility, not independent generalization.
The fixed200-call budget did not reach the gradient-stationarity criterion.
The original2% risk/support rules and unsupported24-group primary are unchanged.
No deployment or submission-readiness claim follows. The next untested model
question is separation of magnitude and decision representations while retaining
the conservative-bound supervision.

Evidence: `moving_full_objective_risk_v1/{analysis.json,result_summary.json,
fitting_error_attribution.json,verification.json,fitting_error_attribution_replay.json}`.
This is native obs8/pred12 image-coordinate risk learning, not a new trajectory
dynamics experiment, SDD raw-frame t50, seconds, metric or foundation evidence.
