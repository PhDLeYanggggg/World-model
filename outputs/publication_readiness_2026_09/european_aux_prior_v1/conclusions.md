# Conclusion: Partial Optimization Repair, No Cost-Transport Promotion

I changed only the auxiliary logit's initial intercept from easy-membership
prevalence to the fitting cap-event prevalence. The 288 new native Torch heads
completed 576,000 updates. Shared/cost initialization, targets, sampler draws,
preprocessing and budget match the frozen controls. Predictions were frozen
in commit9eb1ca68 before any new source-held readout.

## What Improved
Full-input fitting easy-harm MSE has 4 positive / 0 negative / 2 overlapping
assignment intervals against the old true auxiliary. Against cost-only the
counts are 3 / 1 / 2. These dependent fitting diagnostics support a partial
optimization benefit, not a complete repair or independent generalization.

On source-held positive-envelope rows, ranking harm presence with predicted
easy-harm cost improves AUROC over cost-only in all six assignment intervals
(point deltas0.01386 to0.07809). Against the matched repaired shuffled arm,
five intervals are positive. This is harm-presence ranking, not auxiliary
cap-event AUROC, cost-magnitude calibration, or trajectory improvement.

## What Failed
The registered primary is easy-harm cost MSE on source-held positive envelopes.
For full inputs, repaired true versus:

| Reference | Positive / negative / overlapping intervals | Six-assignment point range |
|---|---|---|
| Cost-only | 1 / 2 / 3 | -12.4120% to+0.9243% |
| Old true auxiliary | 2 / 0 / 4 | -10.7484% to+5.6459% |
| Matched repaired shuffled | 0 / 1 / 5 | -11.8103% to+0.0138% |

These ranges are not pooled improvements or confidence intervals. The sole
positive cost-only comparison is P1/C0:+0.9243%,CI[+0.0401%,+1.8086%].
Negative intervals remain at P0/C2:-7.9805%,CI[-15.9370%,-0.0241%],and
P2/C1:-11.5578%,CI[-22.5202%,-0.5954%]. All six results remain in results.md.
All-harm MSE also has a negative guard interval against cost-only and old true.
Motion-only has0 positive /3 negative /3 overlapping primary intervals against
cost-only. It does not rescue the full-family result.

Primary,combined guards and true-versus-shuffled cost gates all fail. The prior
mismatch is not sufficient to explain or fix the transport failure. Better
ranking and lower fitting error cannot replace the failed magnitude objective.

## Evidence Boundary
fresh_run:new training,144 source-held views,1440 direct MSE checks and3000
paired resamples of four localities per assignment after averaging three seeds.
cached_verified:old models,forecasts,features,labels and exactly replayed old
metrics. not_run:new policy/trajectory evaluation and independent selection,
reserved calibration or confirmation. The six assignments overlap and the
source-development localities were already exposed. Intervals are descriptive
and not multiplicity-adjusted. This is not a new independent replication.

Deployment is unchanged. M3W is not submission-ready and the overall research
goal remains active. Detector pixels/native annotation steps only; no metric,
seconds-level,physical-safety,human-gold,true3D or foundation claim.
Stage5C and SMC remain off.
