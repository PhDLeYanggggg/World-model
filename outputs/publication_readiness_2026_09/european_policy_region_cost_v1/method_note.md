# Empirical Objective and Its Limits

Let a_T(x) be the old,frozen causal policy. For a fitting recording g,define
R_g as its observed-label training rows with a_T(x)>=0. The teacher receives
only causal features;future labels define supervised costs,not its action.
For component j in H,R,ER,EH,precompute mu_gj as the mean fitting cost divided
by the existing source cost scale. Record identity is not a model input.

On an unchanged minibatch,estimate the current model's mean p_gj at the
teacher's chosen action among sampled members of R_g. Normalize both by the
original physical-fitting RMS_j. Add the mean generalized I-divergence
between mu_gj and p_gj to the existing continuous/ranking loss,with fixed
weight1. Unknown outcomes and empty regions supply no centroid or safe label.

This explicitly differs from individual-point deviance:the unit is a cost
moment in a fixed policy-selected recording region,not each rare event label.
It also differs from multiplying outputs after training. The model weights
learn from raw398-dimensional causal features,and the original action rule
is unchanged. Original all-action regression remains active.

For epsilon0and positive p,the derivative of
D(t,p)=p-t+t*log(t/p) with respect to log(p) is p-t.
The implemented epsilon1e-12is numerical protection,not a risk threshold.
This gradient identity is elementary,not a new theorem. The synthetic small-
mass test verifies a useful gradient,not a real-world causal failure explanation.

The stochastic group average has variance and a convex-loss shrinkage effect;
it is not an unbiased estimate of every full-record divergence. Large teacher
regions are represented more often. The teacher was fitted on the same training
records,so this is regularized supervised fitting,not independent calibration.
The learned policy can select outside those fixed regions,and physical costs
can be misspecified. All of these can cause another failure. Only the fixed
full-policy/unknown-completion/paired readout can determine advancement.

No independent evaluation role,scientific risk definition,architecture,data
role,normalizer,inference input or trajectory producer is changed.
