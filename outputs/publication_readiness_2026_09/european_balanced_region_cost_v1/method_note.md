# Complete-Region Gradient, Not A New Predictor

Let I_g be a fixed old-teacher selected,known-label fitting region and let
t_gk be its mean supervised cost. Current normalized mean prediction is
p_gk=(sum_{i in I_g}p_theta(i,k)/|I_g|)/s_k,with the same fitting-only RMS s_k.
The regional loss averages D(t_gk/s_k,p_gk) over H,R,ER,EH. Here
D(t,p)=p+epsilon-t+xlogy(t,t)-t log(p+epsilon),epsilon=1e-12.

For one complete region,the derivative coefficient for each row and cost is

```text
(1 - (t_gk/s_k)/(p_gk + epsilon)) / (4 * s_k * |I_g|).
```

First compute the regional prediction means without retaining a graph. Keep
weights unchanged,then accumulate this coefficient times each chunk's output
Jacobian. Add to the ordinary128-row cost/ranking gradient before one common
clip and AdamW step. This reproduces the complete region's chain rule up to
floating-point accumulation;it is not a finite-difference approximation.
Synthetic direct-autograd comparisons and real interruption recovery check
different properties,and both are required.

Only one region is processed at each update,in a fixed seeded cyclic order.
This is not an exact all-regions gradient at every step. Clipping,AdamW,
nonconvexity and within-cycle parameter changes preclude a convergence or
unbiased-clipped-gradient claim. Equal cycles also change the old minibatch
representation weights. The exact mechanism under test is better-defined
regional optimization,not statistical calibration or novelty of the chain rule.

The extra forward visits include all known teacher-region rows in two passes.
They are reported separately from the unchanged base sampler and update count.
Future outcomes are supervised training costs only;region membership metadata
and labels never enter the inference model. This changes neither the frozen
trajectory forecaster nor the fixed positive-gain/predicted2%inference rule.

Generalization and safety must be evaluated separately on the entire required
recording roster. Known-only fitting success cannot resolve unknown outcomes,
new decision regions or independent-domain guarantees. See the
[theory boundary](theory_boundary.md);no metric/seconds,true3D,foundation,
Stage5C execution orSMC claim follows.
