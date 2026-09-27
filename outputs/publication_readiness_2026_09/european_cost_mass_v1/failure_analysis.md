# Failure Analysis

## 1. The Fitting Tradeoff Is Real

The original L2 readout is a restricted origin fit followed by a nested cap.
It can lower squared error by suppressing predictions on the many zero-target
rows while losing predicted harm mass. Full cost-only fitting demonstrates
this jointly in60/72 views; true auxiliary63/72; shuffled59/72. For motion-only
the counts are64/66/65. These are dependent fitting views, not independent
trial counts or a causal proof of the entire neural failure.

Zero H_easy includes non-easy rows as well as harmless easy rows. The94.33%
full cost-only median zero-target share refers to the origin-fit denominator
E[p^2], not to the fraction of observations or a measured easy-degradation rate.

## 2. Projection Is Not the Whole Explanation

The full cost-only median raw mass ratio0.551 already falls to0.199 before
projection, then0.118 after it. Therefore an explanation based only on final
clipping is incomplete. Per-view unprojected and projected diagnostics, not
differences between medians, are available in the private fitting receipts.
Squared error remains a valid mean score; restricted estimation and transport
are the issue tested here. See the primary theory reference in protocol.md.

## 3. Enforcing a Mean Does Not Repair Allocation

The fresh readout changes magnitudes under the same causal caps without adding
causal information. It matches765/864 moment constraints and improves some
coverage-log intervals, yet increases primary MSE. All99 unmatched constraints
are H_easy and hit the preregistered bound8. A bound hit is a failed capacity
constraint for this readout, not permission to increase the bound post hoc.

Full P0/C2 illustrates scene-dependent transport after three-seed averaging:
locality020 still has predicted/actual harm coverage0.0227 under mass, whereas
locality048 reaches4.792 and locality067 reaches16.748. A training-wide mean
constraint can underpredict one held locality and greatly overpredict another.
These are descriptive examples. The complete864-row absolute table includes
every locality and method, not just the largest failures.

## 4. Ranking and Magnitude Remain Different Outcomes

True auxiliary retains positive event-AUROC contrasts but not uniform cost
gains over cost-only or shuffled controls. This preserves a limited ranking
observation without claiming a useful calibrated controller. Full/motion
forecast families also have different event populations and cannot establish
a matched scene-feature ablation. Fitting/outer easy definitions remain the
same ones registered in the parent; this study does not identify every source
of transport mismatch or overturn the mixed crossed-cut findings.

## 5. Engineering Issues Were Separate

A NumPy count-type JSON error was reproduced and repaired through a documented
serialization-only adapter. Exact equality tests cover it. No frozen source,
coefficient, prediction, target, contrast or gate changed. An initial plot call
preceded completion of its aggregate input and exited without output; it was
rerun after the report completed. Neither issue is a numerical model failure.

The negative scientific result remains negative after these operational repairs.
No trajectory intervention, independent confirmation, physical safety, metric
scale or seconds-level horizon was evaluated. Stage5C and SMC remain off.
