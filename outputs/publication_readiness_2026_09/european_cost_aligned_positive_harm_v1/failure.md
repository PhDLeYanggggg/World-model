# Pilot Stopped: Optimization, Not a Scientific Null Result

Registered experiment fc156159. PID47959 exited1 in the first source head,
tree0, before a complete cost model or new validation prediction was available.
The cached controls had already replayed. The128-step Gauss-Newton solver left
gradient0.000160374, above the unchanged1e-7 criterion.

A TRAIN-only diagnostic reused the same106,960 known training rows. Raising
the cap to512 and2048 did not converge (gradients2.63e-6 and1.89e-6). Neither
probe evaluated validation data or changed the loss, initialization or budget.
The new exact-residual-curvature solver passed an independent finite-difference
test and solved that same tree in7 iterations,0.744s, gradient9.97e-8.

The original code and registration remain unchanged. Continue under the
separately registered european_cost_harm_newton_v1 amendment. These are
engineering results, not evidence the model improves prediction or selection.
Stage5C/SMC off; independent roles closed; no deployment promotion.
