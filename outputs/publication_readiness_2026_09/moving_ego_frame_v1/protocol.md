# Ego-History Frame: Fixed Causal Representation Control

Motivation is the completed negative forecast-tail probe, not a new test readout.
Its source motion ranges differ markedly. The original scale is maximum1pixel,
ego-history radius and valid-neighbor-history radius. This is causal, but gives
outer neighbor geometry a role in rescaling ego motion. A representation repair
is a hypothesis, not an already established cause or new scientific metric.

Keep signed all/easy targets, their original causal-scaled units, weights,
source exclusions, normalization fitting boundaries and zero decision threshold.
Reuse original120 signed-risk ExtraTrees controls. Fit two fixed arms per source:
history120 recomputes the existing64+32+24 features after dividing all observed
xy and predicted floor/candidate coordinates by the ego-history scale instead;
history121 additionally appends log(ego_scale/original_scale) to retain context.
Time and valid masks are unchanged. Ego scale=max(1pixel,max past ego radius),
using the existing current-relative observation frame. No future, outcome mask,
source ID or action ID enters inference. The one-image-unit floor is explicit;
it is not a claim of full coordinate/rotation invariance or physical calibration.

Sixteen new forests and eight cached controls over the same8source-held folds.
Same64trees,depth8,minleaf32,seed17,maxfeatures1,bootstrapfalse,n_jobs4. No target
or target-RMS change. Input preprocessing uses fitting sources only. Primary
history121-minus-original120 includes one additional causal scalar; history120
is a secondary ablation of retaining that scale context, not a replaceable primary.
This is a conventional representation test, not a novel forest or new dynamics.

Fixed information screen: >=1% original joint decision-cost reduction and
nonworse false-safe cost, in>=2/3contexts. Report normalized riskMSE and all
source counts, including empty admissions. This screen is not original safety
certification. No independent role, C/H or reserved-test readout is opened.
Original2% all/easy positive-harm, easy-net, support and empty-control NA remain.
No identity exception, amplitude/threshold sweep, model deployment, Stage5C/SMC.
The original24-group primary remains NA. Outcomes are partial available labels;
unknown futures stay unknown. Three contexts from one producer seed are not
independent confirmation. Prior negative target/tail/amplitude studies stay failed.

Commit inputs, protocol and code before fitting. Real pilot, per-cell atomic
checkpoints, explicit resume, heartbeat and full refits are required. Local
CPU4/workers0 suffices based on the completed same-size probes. If this contrast
fails, do not expand it into neural retraining or use its best source as a win.
