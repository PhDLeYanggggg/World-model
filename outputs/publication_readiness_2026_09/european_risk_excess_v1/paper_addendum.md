# Development Ablation: Risk-Objective Alignment Is Candidate-Dependent

We retain a frozen trajectory bank and matched controller architecture,
initialization, normalized inputs, sampling and update budget, changing only
two-moment regression to regression of signed excess over a2% relative-harm
budget.144 fresh heads span three seeds and source-excluded controller fits.

For the neural candidate, fitting signed-score MSE improves27.25%, but the
held-source contrast is-3.06% [95% exploratory locality interval:-12.32,3.91].
The paired diagnostic trajectory-gain change is0.23percentage points
[-0.18,0.77]. Neither supports a neural improvement. In contrast, fixed damping
shows7.18% [2.65,12.27] lower held-score MSE and2.38points [1.65,3.31] added
diagnostic ADE gain relative to CV. All three damping seed MSE intervals are
positive. The control improvement is retained rather than attributing it to
the neural trajectory model.

The neural risk-only screen retains3.52% [2.49,4.64] positive harm/reference
error on its accepted subset. Although its locality-mean easy gain is positive,
the worst dependent role/seed/locality view degrades easy ADE by5.61%. These
screens omit the full stationary/utility/easy policy and do not certify safety.

Intervals use3000 locality bootstrap draws after averaging dependent source/seed
views; independent roles remain closed. Results are source-development ablations,
not submission-ready confirmation or a novel-method claim. The next test must
separate fitting, calibration and held-source evaluation across the whole model
chain. No deployable neural world-dynamics contribution follows from this ablation.
