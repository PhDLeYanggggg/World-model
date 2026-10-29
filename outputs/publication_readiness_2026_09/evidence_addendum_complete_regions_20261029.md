# Manuscript Evidence Addendum: Complete-Region Optimization

This extends the frozen revision7draft and seventh-study addendum without
rewriting them. Eight controlled studies now comprise432actual downstream
controller heads and702,000primary updates,not new full trajectory predictors.

## Method And Hypothesis

The previous frozen-state diagnosis found selection-conditioned easy-harm
underprediction despite better average cost prediction. A subsequent54-state
gradient check measured a substantial difference between represented-minibatch
and complete-region moment gradients;some teacher-selected regions never
received the auxiliary loss across2,000steps. This finite diagnostic includes
Monte Carlo variability and different implicit region weights. It is not proof
of a unique asymptotic bias or the complete cause of policy failure.

We tested cyclic complete-region gradients while matching architecture,inputs,
initialization,base sampler,teacher regions,normalization,optimizer,schedule
and inference. Each update adds the exact within-region chain rule in chunks
before clipping and stepping. The changed region weighting and additional
12,275,011teacher-row visits are explicit;the comparison is not equal-FLOPs.
All54new endpoints were committed before the fixed exposed inner-held readout.

## Result

The mixed policy's known-label ADE gain over the floor was0.943728%,compared
with0.943808%for the stochastic-region controller. Their paired difference
was-0.000080percentage points,conditional95%locality interval
[-0.108174,+0.097264]. The mixed-minus-physical difference was+0.011277points
([-0.121830,+0.102166]);missing-outcome completion bounds did not establish
positive neural value. Required-risk support improved from19to24of186repeated
record/head views,but complete protection remained far from satisfied. Known
easy violations fell89to76without a deployment pass.

The fixed teacher region's median EH prediction/observation ratio was0.112670,
versus0.132688previously. Its own new selected region had ratio0.039726.
Thus correcting within-region gradient calculation did not solve the immediate
cost-fitting goal or the downstream policy problem. The result limits the
gradient-defect hypothesis rather than demonstrating a new calibration method.

## Claim Boundary And Next Discriminating Experiment

The study is a negative controlled optimization result,not independent
confirmation,a neural-world-dynamics improvement or a calibration theorem.
Its3,000-locality intervals are conditional on one frozen forecasting producer;
three controller seeds are not three full pipeline seeds. SOURCE development
was already exposed. Unknown outcomes,undefined selected risk and complete
recording requirements are not replaced by known-label averages.

The next diagnostic measures neural marginal candidate opportunity above a
physical per-query oracle before another cost-loss trial. Such an oracle uses
future labels and can only bound a saved-candidate class,not establish causal
learnability or deployed performance. Independent evidence,strong forecasting
controls,scene-joint contribution and main-method novelty remain unresolved.
No submission-ready,true3D,foundation,metric/seconds,Stage5C orSMC claim follows.
