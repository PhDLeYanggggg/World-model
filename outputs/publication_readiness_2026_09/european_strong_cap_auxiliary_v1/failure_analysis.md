# Failure Analysis

## Evidence-Supported Findings
1. Implementation drift in the base model is ruled out for this comparison:
   all144 original controls reconstruct bit-exactly with identical samples,
   input/target hashes, preprocessing, architecture and four-cost objective.
2. The fixed auxiliary is not a consistent expected-cost improvement. Full
   primary intervals are1 positive/1 negative/4 overlap versus original and
   0/2/4 versus shuffled. Motion primary is0/1/5 versus original and0/2/4
   versus shuffled. No favorable family or assignment is selected.
3. Failure already appears on fitting rows:52/72 full-input fitting primary
   comparisons deteriorate.29 also deteriorate held;14 improve fitting but
   deteriorate held. Against shuffled,28 fitting-positive comparisons become
   held-negative. These dependent counts are descriptive, not extra tests.
4. Most fixed-batch cost losses decrease:139/144 cost-only,138/144 true-aux,
   139/144 shuffled. Their final medians are0.44553,0.46088,0.45202. This is
   optimization progress, not whole-fitting MSE or out-of-sample success.
5. Event probability remains learnable without reliable magnitude gain.
   Full cap AUROC median0.84617,69/72 log losses better than fitting prior.
   Motion median0.61753,33/72 better,with3 unsupported ranking views.

## Largest Descriptive Failure Slices
Full P0/C2's four-locality three-seed primary gain is-14.8538%,with adverse
95% interval[-29.1706,-0.5371]. Within that assignment,locality067 has a
three-seed mean-38.0483%,and locality048-20.2928%. The worst individual
full seed/locality view is P0/C2/seed43/locality067:-93.9549%. These are
cost-MSE percentages, not trajectory degradation or physical harm rates.
They are retained as diagnostics, not excluded or used to tune a new model.

## Hypotheses Not Yet Established
Shared-gradient interference, unequal task scale, finite fixed training
budget and locality-dependent magnitude shift remain plausible. No gradient
conflict measurement or alternate-weight intervention was performed, so none
is declared the cause. Inner teachers trained on two localities and outer
teachers trained on three still define potentially different cap events.
Rare support is especially weak in some motion views. Neither scarcity nor
teacher-size mismatch was isolated by this experiment.

The old architecture/objective/support mismatch cannot explain away this
new matched negative result. It also does not follow that any cap-event
auxiliary must fail, or that more epochs automatically repair it.

## Next Falsifiable Step
Keep the strong model fixed. On fitting/inner-held data only, measure
cost/auxiliary gradient alignment and separate early from late fitting
cost changes, retaining true/shuffled controls and all source assignments.
Test a budget or loss-scheduling repair only after that diagnostic supports
it, with a preregistered matrix and no outer-outcome weight selection.
If fitting interference is not supported, prioritize producer/event and
magnitude transport rather than another arbitrary architecture sweep.

Keep independent selection/calibration/confirmation closed and deployment
unchanged. No new trajectory or policy gain is established. Native-step
pixel-space only; Stage5C/SMC remain off.
