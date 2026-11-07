# Train Against Unobserved-Row Disagreement

The matched easy-only continuation lowers known easy harm but loses 7.68 pp
on the held paired completion lower and does not expand candidate feasibility.
This follow-up adds one training term. On wholly unobserved fitting rows,
penalize the maximum displacement of the forecast from causal CV over the
12 requested steps. The term is not a surrogate ground-truth position.
Partially observed rows retain the existing masked ADE and easy loss. Neither
outcome mask nor any future position is an inference input.

Use the original PCT fold-2 4,000-step checkpoints, optimizer states, 300
continuation steps, fitting-only cohort, 64-row draws and LR of 1e-4. The
new objective is the frozen easy-only objective plus one native-scale radius
term using the original fitting factors. No search over its coefficient.
Train three actual seeds [17,29,43]. Reuse the existing 300-step ADE and
easy-only controls after hash, draw stream and RNG verification. No duplicate
control fitting. Do not start from the final easy-only model.

Freeze all three new full A forecasts before accessing A labels. Compare
directly with each matched control: known utility, paired completion-lower
utility, known easy harm and hard utility. Measure change in the unknown
radius to CV separately to check the intended mechanism. A lower radius is
not a policy pass. Retain original full recording 2%/support/NA checks for
the raw path and the exact free-count label-aware expanded-bank ceiling. A
valid methodological result must not be defined solely by smaller movement.
Use all 125,540 A rows and all three seeds; 3,000 paired source draws across
14 exposed development sources. The held easy/hard cuts are unchanged.

No proxy labels, input leakage, identity certification, new residual head,
external outcome access, role change, deployment, Stage5C or SMC. Native
8/12 image coordinates. This is a targeted continuation experiment, not
independent confirmation or a from-scratch foundation training claim.
