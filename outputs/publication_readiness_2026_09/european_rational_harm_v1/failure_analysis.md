# The Local Gradient Repair Worked; Safe Selection Did Not

## Completed Evidence

All18 matched fits and36,000 updates completed, followed by144 fixed readouts
against seven frozen controls. Real initial objectives, initial parameters,
source/query draws, protected outputs, final identities and control predictions
reproduce. Additional72 fit-only autograd views completed64.016528s, PID53106
exit0. No independent data, threshold selection or deployment change occurred.

## The Targeted Mechanism Was Actually Changed

On the identical58,707 full-range-selected fitting occurrences, the hard-clipped
head has54,254 exact-zero easy-harm Jacobians (92.4149%); the rational head has
231 (0.3935%). Positive easy-harm underprediction allocated to exact-zero J
falls from76.590339 to0.001224. This is genuine local gradient repair, not merely
a function implemented on synthetic inputs. It does not establish a useful
parameter gradient, convergence, or generalization.

Near-zero derivatives remain:40,174/58,707 (68.4314%) on that fixed cohort under
the registered1e-6-cap definition. Positive easy-harm underprediction barely
changes:79.541375 to78.718189. Increasing differentiability does not ensure that
gradients identify the rare rows that harm easy cases.

## Selection-Conditional Failure Remains

The rational model's own48,990 fitting occurrences include1,265 unknown outcomes.
Among known outcomes, predicted easy harm totals0.744652 versus observed69.357583
(1.0736%). On the full eligible cohort, it predicts15,795.331832 versus observed
5,060.485293, so global overprediction coexists with severe own-selected
underprediction. These are repeated source-normalized aggregate moments, not
calibrated probabilities, independent n, causal effects or ADE/FDE.

37,758/48,990 own-selected occurrences (77.0729%) have near-zero J;87.6003% of
positive underprediction lies there. Exact-zero alone is no longer the dominant
mechanism. This supports investigating small-gradient conditioning and
policy-conditional estimation, but does not prove feature insufficiency or
identify optimization as the only remaining cause.

Logged preclip norms improve relative to hard clipping (median44.0252 vs84.3771,
median clip factor0.11357 vs0.05926), but remain worse than isolated odds
(9.8881,0.50566).97.62% of logged batches exceed clip5 in both wider branches.
These378 logged batches per arm are not every update or a convergence proof.

## Fixed Outcome Gate Failed

Held rational-minus-hard-clipped utility at matched intervention counts is
-0.217080pp[-0.376888,-0.062234]; at full counts,-1.547315pp[-2.429844,-0.825328].
Versus isolated odds, matched+0.201973[-0.029673,0.473714] is not stable positive.
Versus joint-risk,+0.058333[-0.101095,0.226338] is also inconclusive. One matched
contrast versus the older risk head is positive,+0.151913[0.032439,0.286865],
but it is a lower-total-budget control and does not replace the complete gate.

Known easy violations decline44 to34 views compared with hard clipping, but
undefined easy views increase12 to19 and selections fall31,270 to27,371.
Full-policy support remains0/72. Therefore fewer known violations do not prove
safer deployment. Held signed MSE6.644551 is lower than hard-clipped7.774976,
with difference CI crossing zero, and worse than isolated5.047722. The registered
advance rule fails14 conditions. No winner is promoted by selecting a favorable
point estimate, comparator, seed, subgroup or undefined denominator.

Intervals are3,000-draw exploratory locality bootstrap over12 exposed original
TRAIN localities. Repeated contexts/head seeds are averaged within locality;
they are not new forecasting producers or independent confirmation. Utility is
pair-specific completion-lower reference-normalized mass, not forecast FDE.

## Failure Taxonomy and Next Action

1. Confirmed numerical/local obstruction: hard-clipped exact-zero Jacobians;
   substantially repaired on real fitting inputs.
2. Remaining conditioning: many tiny derivatives and strong gradient clipping;
   magnitude measured, sole-cause explanation unproven.
3. Selection-conditional estimation: global overprediction but severe
   underprediction in the chosen cohort; not fixed by a wider/smoother head.
4. Evidence/support: missing outcomes and undefined references still prevent
   full risk support. Do not drop them or count empty policies as safe success.
5. Generalization: no independent calibration/test claim available yet; exposed
   internal recordings cannot be renamed to restore independence.

The next smallest useful experiment is a producer/recording-excluded,
policy-conditional risk-calibration comparison inside original TRAIN, keeping
the utility predictor frozen and the2% budget unchanged. First verify support
and teacher-exclusion identities for every proposed fitting/calibration fold;
retain the raw, globally adjusted and matched-coverage controls. Do not start
another ad hoc bounded mapping or select an offset on the present held results.
Independent roles stay sealed until the fixed mechanism is sufficiently justified.
This is the next task, not an executed calibration result.

Obs8/pred12 raw stride12, detector-track pixel/dataset-local coordinates only.
No metric, seconds, human gold, physical safety, true3D or foundation claim.
Stage5C and SMC remain disabled. No deployable or submission-candidate promotion.
