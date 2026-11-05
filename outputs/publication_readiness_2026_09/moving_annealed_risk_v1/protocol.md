# Fixed-Schedule Neural Risk Continuation

## Question

The complete-B diagnostic finds aligned all/easy shared gradients in all nine
checkpoint views. Projection is identical to the ordinary update, so I do not
train a projected-gradient variant. At two of three original MSE checkpoints,
the saved-AdamW joint step increases full-fitting easy MSE despite a favorable
raw gradient direction. This is a local diagnostic, not proof that optimizer
overshoot causes held-scene failure. The fixed learning rate and final-iterate
fit are the next testable factors.

## Matched Model Change

Train six real neural continuations: three original own-B contexts, each with
constant versus cosine learning rate. Both start at identical warm kinematic
weights, reset AdamW identically, and use the original unmasked 492 inputs,
32,130 parameters, MSE, bound-aware unknown labels, query sampling and 2,000
updates. The sole scheduled change is the step size, from .001 to .00001 over
the fixed full budget. It is not a parameter search or a novel method claim.
AdamW's effective decay step also changes with learning rate; do not claim
to isolate stochastic momentum from all other optimizer effects.

Use the final iterate, never a V-selected checkpoint. Constant controls must
exactly reproduce the earlier MSE weights and draw hashes. Fitting-only full
MSE and sign-error diagnostics are saved for both arms before V readout.
All six fits freeze before any new prediction on V. The first cosine fit is
repeated for exact reproducibility. Both arms receive an actual 50+50 versus
100-update resume test, with the schedule still defined over 2,000 updates.

## Evaluation

Use the same original seven exposed SOURCE units and 17 recording views in
nested C. Fixed proposals, rules, .02 all/easy budgets and support remain.
Primary contrast: cosine minus constant. Also report warm and affine controls.
No threshold search, new H, external reserved data, identity certificate or
conversion of empty controls to zero. Previous failures remain failures.

The inherited development expansion screen requires positive missing-label
completion lower contrast in all three contexts, a nonempty intervention,
no increase in each context's known all/easy violations or undefined selected-
easy/full-easy support, and positive aggregate completion lower versus affine.
This screen is not deployment certification. The original full-policy safety
and support gate still requires every recording view, and the original
24-group primary remains not estimable, not passed.

Report all contexts and failure slices. Completion bounds are not sampling CIs.
One producer seed and repeated development observations are not independent
confirmation. Any later expansion requires this experiment's evidence first.

## Runtime And Scope

Native arm64 Torch, CPU4/inter-op1/workers0; checkpoint and heartbeat every100
updates, exact resume, 12h boundary, >=5GiB free disk and <=512MiB private
artifacts. Checkpoints and row caches stay private. Training on the unchanged
causal risk-head schema is not new trajectory dynamics or JEPA pretraining.
Native observation8/prediction12 image pixels, no verified seconds, metric,
true3D, foundation or human-gold claims. Stage5C and SMC remain disabled.
