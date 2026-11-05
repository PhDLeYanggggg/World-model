# Preflight Test Correction

Before any real-data pilot or primary fitting, the first synthetic test run
failed three assertions. Two mixed a preprocessing-isolation test with an
unjustified assumption that a shallow random forest must accept candidates in
a small continuous-feature toy dataset. The third used zero absolute tolerance
when comparing two weighted summations, rejecting a7.4e-17rounding difference.

Isolation remains an exact parameter-equality test after changing held features
and labels. Both-decision learning is now tested separately with a genuinely
separable binary signal under the same fixed forest capacity. Weighted mean
comparison uses1e-12absolute arithmetic tolerance. Real model settings, targets,
features, partitions, scientific thresholds and the prospective screen do not
change. No real fitted result was consulted for this correction.

The preliminary registration completed while that test run was pending and is
retained as registration.json. It authorized no completed fits. The active
registration_v2.json binds corrected tests and this note before training. This
is an engineering test correction, not a successful scientific experiment.
