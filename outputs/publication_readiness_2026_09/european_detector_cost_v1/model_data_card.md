# Detector-History Cost Heads: Model And Data Card

## Intended Use

A fitting-only matched input ablation for the question of when a frozen neural
forecast should replace a strong causal motion floor. This is not a newly
trained trajectory generator, JEPA representation, dynamics model or deployment
package. The primary is history plus the physical/neural candidate bank.
Summary, gates and exact replay determine the result; secondary controls cannot
replace the registered primary after seeing scores.

## Data And Labels

European SOURCE uses supplied detector-track box midpoints in pixels, with
eight observations and twelve prediction steps separated by twelve raw frames.
74 fitting recordings provide149,510 unique packed queries. Repeated contexts,
controller seeds and overlapping windows are not new independent observations.
Independent selection/calibration/confirmation, held readout and DroneCrowd are
not admitted. Data role here is source supervised fitting and diagnostic-only
resubstitution, not official independent evaluation.

The additional64features derive exclusively from eight supplied past/current
boxes and raw detector confidences. Raw source keys, boxes, midpoints and width
match the original packed inputs exactly, with222real prefix-truncation checks
and a separate complete raw reparse. Confidence is a detector output, not a
calibrated probability of trajectory correctness. Sensor-acquisition-as-of of
the tracker export is not certified. Labels are detector-silver, not human gold.

Unknown future outcome rows remain unknown in scoring. They may occur in the
unchanged causal training sampler but receive no known-only supervised loss.
Targets include benefit,harm,reference,easy reference,easy harm; no future
endpoint, future-support mask, remaining track length, central velocity or
test endpoint goal becomes an inference input. Extra normalizers use each
own-context fitting input set without conditioning on future label support.

## Model And Controls

All144heads share width64,30,405parameters,Adam,2,000updates,batch128,
learning rate0.001,gradient clipping5,asymmetric under-harm weight4 and
ranking weight0.5. A zero-initial64x64input branch supplements the original
398fields. Four arms are disabled,current-only,key-noise and full history.
Current/noise have matched parameter counts, not matched information ranks.
The noise control can carry identity; it is not a visual modality.

Six original fitting contexts, controller seeds17/29/43 and two candidate banks
are retained. The trajectory producer remains frozen seed43. All36disabled
2,000-step endpoints reproduce their old model/Adam/draw/RNG/trace states
exactly. The first complete history/mixed refit is exact, excluding elapsed
time metadata. Training is actual native-arm64 Torch CPU4/workers0, not a
NumPy fallback. No new CREATE job is required by this measured resource use.

## Safety And Evidence Limits

The original2%complete all/easy recording risk contract remains. Empty risk
is not_estimable; same-count controls test utility only. Missing-outcome
completion mass remains in utility/risk bounds. Twelve-locality bootstrap
averages repeated contexts/head seeds first and uses3,000resamples. These
intervals describe fitting evidence, not independent generalization or
whole-pipeline OOF. Minibatch loss and all-action moment MSE do not certify
calibration after action selection. No forecast error is measured in meters
or seconds; homography/scale are not validated by this feature change.

See readout.md,evidence.json,training_traces.json and failure_analysis.md for
the measured outcome. Raw data,private feature arrays,images and checkpoints
remain local. Deployment is unchanged. No Stage5C execution,SMC,true3D or
foundation-world-model claim.
