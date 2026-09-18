# Fixed Direction Null: Diagnostic Scope

Written during the motion-loss experiment after early unprotected negative
scores were observed, before computing direction-null costs. Explicitly post-hoc,
not part of the original pre-fit analysis and not independent confirmation.

For both complete twelve-model families, evaluate each fixed relative prediction
and its +90-degree, -90-degree and180-degree rotations about the current origin.
Preserve each step's displacement magnitude and trajectory shape under rotation.
Keep all rows, all seeds and all four source folds. No label chooses a rotation,
no new model is trained, and no outer/main role is scored. These are numerical
negative controls, not new generative proposals or deployed trajectory choices.

Report actual equal-site ADE gain and each arm's CV/candidate future-oracle gain.
Compare original-minus-null oracle gain with the same2000 conditional site draws
and seed38113. All three contrasts are shown, with no preferred null selected.
The intervals are descriptive, not multiplicity-adjusted confirmation.

Question: does better oracle utility depend on the predicted direction, or could
matched-length arbitrary directions provide similar oracle space? A positive
contrast would support direction alignment against these limited nulls, not
causal feature sufficiency, a learned safe gate, physical consistency or novelty.
Equal zero-target harm is a numerical invariant, not a passed safety gate.
Future targets are used only to score the fixed arrays. Pixels/raw frames only;
no metric/seconds claim. Stage5C execution and SMC remain off.
