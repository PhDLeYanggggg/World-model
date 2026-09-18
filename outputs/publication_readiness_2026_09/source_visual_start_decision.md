# Source-Supported Visual Start Information

The previous45-fit geometry study found no bidirectional start-probability gain.
Positive Hotel contrasts were explained by mean-probability shift and all source/
mixed models lost to a constant source prior. The next falsifiable question is
whether actual past pixels add sample-specific predictive value beyond the
same geometry and coverage information, rather than repeating a routing sweep.

## Fixed Contrast

Two arms: geometry+coverage-only and geometry+pastRGB+coverage. Both instantiate
the same small shared frame CNN and fusion MLP, with identical initialization,
weighted training-row sequence, optimizer and budget. Coverage-only zeroes RGB
before its four-channel encoder; unsupported pixels and wholly unavailable frames
are masked in both arms. Eight past32x32 cached crops are kept in temporal order.
No quality filtering or new row membership; missing visual support is reported.

Schedules: main-only,source-only,mixed(one-half sample mass per domain).
Seeds17/29/43. Two exposed main held-site directions; source-only fits are shared
between them. This gives30fresh Torch models and36prediction cells. Each model
uses2,000updates,batch64,AdamW learning-rate0.0003,weight-decay0.0001,gradient-clip5.
Checkpoints every200updates. Final fixed-budget checkpoint,not held-best selection.
An initial100-update training-only pilot measures local feasibility and resumes
within the same budget. CPU4/inter-op1/workers0;no Rosetta or resource probing.

## Inputs and Labels

Reuse the approved original SDD train40 source,geometry/image manifests and the
previous stationary query set. Source22,374complete-label stationary queries;
6,460incomplete stationary queries are counted as unscored,not negative examples.
Main365stationary fit queries/31local IDs in ETH and Hotel;Zara has no such
queries and is not_run. Native annotation steps and source+144rawframes are not
physically time-equated. An exact later annotation-coordinate change is a silver
supervisory proxy,not human intent or a body-motion gold label.

Geometry uses the frozen476-column observed-unit v2 schema. Training statistics
use only each arm's training inputs/weights;source-only uses no main normalization
or fitted labels. Images use fixed uint8 scaling and cached past-only correspondence.
No endpoint,central velocity,test goals,teacher predictions,future pixels or
source provenance IDs enter the estimator. Future targets supply loss/evaluation
labels only. No change to the main11,966forecast cohort,equal-site past-normalized
ADE or closed development/calibration/confirmation roles.

## Analysis and Claim Boundary

Primary information contrast: Brier lift of RGB over the same-schedule mask arm,
reported separately by held site and seed. Also report both arms versus opposite
main-site training prior,their own weighted training prior,and source constant
prior. Training priors are computed without held labels. AUROC/AUPRC/ECE/log loss
are secondary. Decompose Brier changes into mean-probability shift and varying
prediction terms as a **descriptive** analysis,not inference recalibration.

Use2,000conditional held-agent bootstrap draws and average seed losses,not
probabilities. Five ETH and26HotelIDs,overlapping windows,contemporaneous agents
and repeated site exposure prevent independent confirmation claims. A useful
visual-information hypothesis needs consistent RGB benefit beyond mask/prior
controls in both directions;even that would not establish trajectory direction,
safe switching,joint intervention or deployment. Report all arms and warnings.

No threshold search,selected ensemble,residual model or new deployment follows
automatically. Do not change thresholds from these exposed held results.
Previous classical/MLP results are context,not matched-budget controls for this
new CNN comparison. Stage5C/SMC remain disabled. No metric,seconds,true3D,
foundation or submission-readiness claim.
