# Loss Alignment Helps Damping, Not a Reliable Neural Controller

## Completed Contrast

144 fresh native Torch heads,288000updates, paired with144 sealed moment-MSE
controls. The architecture, initialization, features, normalization, source
roles, seeds, draw streams and training budget match. Only the objective changes
to squared error of positive harm minus0.02CV ADE. Control inference and sampling
match exactly. Predictions were frozen before this readout. No new trajectory
forecaster, independent-role readout or threshold selection was introduced.

## Main Result

The primary neural held-source signed-MSE gain is **-3.057% [-12.321,3.908]**.
It does not establish improvement; the interval also does not establish a
uniform deterioration. Fitting MSE improves27.25% [22.83,31.42], but none of
the three neural seed intervals has a positive lower bound on held sources.
Six locality point estimates improve, six worsen. The largest deterioration,
locality124, is48.24% worse MSE. It remains in every aggregate and interval.

Neural diagnostic coverage rises21.98% to24.98%. Its all-row net ADE gain vs CV
rises0.859% to1.090%, but the paired change is only0.231percentage points
[-0.180,0.768]. This is not a demonstrated trajectory-selection improvement.
Screened positive harm falls in point estimate from4.867% to3.523%
[2.485,4.642], still above2%; nine locality means exceed2%. Fitting-source
screened harm remains2.850%. Changing the loss alone has not calibrated the
decision subset even inside the fitting sources.

## Successful Control and Its Limits

For fixed damping, held signed-MSE improves **7.183% [2.646,12.267]**, with
positive lower intervals for all three seeds. Its diagnostic net ADE gain vs
CV rises2.226% to4.604%. The paired increase is2.378percentage points
[1.651,3.314], positive in all twelve locality means. This useful causal
control result must not be hidden to make the neural candidate look stronger.

Damping is still not certified for deployment: average screened positive harm
is1.936% [1.073,3.206], three locality means exceed2%, and locality020 reaches
7.455%. A mean below2% is not a worst-source guarantee. This is an all-risk-only
diagnostic without the original complete safety head or scene-joint control.

## Easy and Zero-Reference Failures

Neural easy ADE improves5.887% on average, with all locality means positive.
However, five of72dependent role/seed/locality views degrade by more than2%;
the worst is **5.614%** in single2/seed29/locality110. Damping has three such
views, worst3.068% in single0/seed17/locality020. The locality-mean gate does
not assert all individual views are safe. [Exact slices](slice_diagnosis.md).

One neural row-view has CV error zero and added image-local ADE0.17435419.
Past-only inspection confirms its last observed displacement is zero. The
existing full-policy stationary guard would reject that case, but this is not
a fresh full-policy evaluation or a safety certificate. The diagnostic deliberately
omits stationary, utility and easy guards for both objectives. Zero-reference
harm has no percentage denominator; positive harm is not net easy degradation.

## Failure Taxonomy and Next Test

- **Objective mismatch, partly addressed:** the direct target learns the fitting
  score substantially better and benefits damping out of source. The loss is
  implemented and trainable; this does not prove neural risk generalization.
- **Source generalization, unresolved:** the neural fitting advantage does not
  survive holdout. This is consistent with overfitting, missing conditional
  support or candidate-specific error variability, not proof of one unique cause.
- **Conditional calibration, unresolved:** selected-set risk is excessive despite
  better fitting. A score sign is not a confidence bound. Do not increase tolerance.
- **Parameterization, possible contributor:** the retained two-component basis is
  unidentifiable under the new scalar loss. Its components must not be interpreted
  as calibrated moments. This experiment does not causally isolate this issue.
- **Safety-floor omission, explicit:** the risk-only screen cannot replace the
  stationary/easy/gain safeguards. Average easy gains hide real individual failures.

The next evidence-producing step is strict source-separated calibration and
support-aware abstention on fixed score families, retaining the stronger damping
control. The whole chain, including forecast producers and score-head fitting,
must exclude the held source. Do not train a meta-calibrator on ordinary cross-fit
outputs whose producers have seen the eventual held-source outcomes. Keep both
objectives as fixed controls rather than selecting a winner on this readout.
Any claim about a complete policy requires a separately registered matched test.

## Claim Boundary

No safe neural advantage is established. Deployment is unchanged. This is
opened-source obs8/pred12 at raw-frame stride12, detector-derived silver
image-local data, not historical Stage37t50 or independent confirmation.
No metric, seconds, physical-safety, true3D or foundation claim. No Stage5C/SMC.
