# Prespecified Fitting-Region Mechanism Check

Recorded before the first primary readout. This is a diagnostic-only comparison,
not an additional model selector,threshold search or independent calibration.
Use all54frozen final endpoints and the same old54scene teachers,only original
fitting recordings. No optimizer updates. Keep all1,332record/head/bank views.

Evaluate each new cost model on two regions:the old teacher's fixed chosen
action,including actions now rejected by the new gate,and the new model's
actual chosen action. Retain unknown outcomes in region counts and undefined
ratios. Report predicted/true H/EH/R/ER mass medians and how many new actions
lie outside the teacher's old region. The latter includes newly selected rows
and different candidates at previously selected rows.

Original teacher/current choices must exactly reproduce the primary fitting
readout hashes. Main policy and control metrics are not changed by this check.
Store full aggregate-record diagnostics in a compressed JSON archive,not raw
trajectories or weights. Recompute in a separate process for exact verification.

Better calibration inside the same fitting region is resubstitution,not
generalization. If the policy moves outside that region,the new selected costs
can still be biased. If fixed-region calibration does not improve,the new
objective did not achieve its immediate empirical goal. Neither case is proof
of a unique causal mechanism or a reason to open independent roles.
