# Conclusion

Nine partial-neighbor neural forecasters completed 4,000 updates each, with
three fixed seeds and matched locality folds. A fresh 4,000-update legacy
control reproduces every model parameter exactly, including after pilot
checkpoint resume. Every new/old pair has identical sampling counts and final
sampler state. All weights and predictions were frozen before comparison.

The primary ADE gain against legacy neural prediction is +0.012%
with a 95% exploratory locality interval [-0.374, +0.412]%. Positive-easy gain
is +0.335%; hard gain is -0.152%.
The preregistered exploratory benefit screen is **failed**.
See results.md for causal-baseline comparisons; improving a weak neural
control alone does not establish useful intervention or safe deployment.

Training is fresh native Torch CPU work, not a NumPy fallback. The nine new
fits used 899.93 training seconds in total, excluding preflight and
inference. Cached legacy assets are hash-verified and replayed with fresh
inference. Raw source labels are detector-derived, not human gold.

The task is eight observations and twelve requested future steps at raw-frame
stride12. It is not raw t+50, calibrated seconds, metric 3D or foundation-model
evidence. Independent selection/calibration/confirmation remain closed. No
new gain/harm policy is trained, no deployment is changed, and Stage5C/SMC
remain disabled. A supported input fix is not a calibrated safety guarantee.
