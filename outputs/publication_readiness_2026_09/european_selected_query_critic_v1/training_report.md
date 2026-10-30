# Real selected-query critic training

## Material Passport

fresh_run:36 full native-arm64 Torch risk fits,72,000 primary gradient updates.
Both first context/seed17 heads have independent2,000-step exact refits (4,000 additional updates).
cached_verified: original forecast/cost heads, causal features, source inputs and selected-action identities.
No newly trained trajectory forecaster, threshold selection or independent outcome readout.
Training PID41637 takes294.916s,peak RSS8783167488B.
All36 heads have30,726 trainable parameters. Row-local and selected-set pairs share initialization,
source-balanced draws, label masks, fitting-only normalizers, architecture size and loss weights.
Benefit,reference and easy-reference predictions are frozen; only harm/easy-harm are corrected.

## Fixed fitting-monitor loss

These are16 fixed source-balanced training queries per head,not validation/test loss or full-dataset means.
Every candidate is final-step2,000; no checkpoint is selected from these curves.

|Arm|Heads|Monitor total decreased|Median initial total|Median final total|Median initial query score|Median final query score|
|---|---:|---:|---:|---:|---:|---:|
|row_local|18|18|0.057300524|0.035913458|0.000435536|0.000443910|
|selected_set|18|18|0.057300524|0.032884933|0.000435536|0.000448284|

## Actual component-gradient telemetry

Norm ratios compare the two unweighted component gradients from the SAME actual update batch.
Their registered weights are both0.5,so the ratio also applies after these weights.
This is not a ratio between medians drawn from different experiments.

|Arm|Snapshots|Query/point norm min,Q1,median,Q3,max|Zero query gradient|Negative cosine|Query gradient below1%|
|---|---:|---|---:|---:|---:|
|row_local|738|[2.9143775668158034e-06, 0.00027478076962213454, 0.0008093232204893755, 0.002679731022646197, 0.40897048563392846]|0|328|679|
|selected_set|738|[1.0211837638367116e-07, 0.0002607414477288258, 0.0007081863900496629, 0.001823012668675231, 0.16722236169607385]|0|335|698|

All requested candidate actions and current-query peers are retained before supervision masks.
Unknown labels contribute no supervised loss. The query target is supported only for nonempty,
fully known original selected subsets; empty/unknown subsets are not zero-risk targets.
Lower fitting loss is optimization evidence,not calibrated post-selection risk or independent lift.
Gradient weakness/conflict is measured but does not by itself identify the unique cause of policy failure.
The new policy still reassigns actions after training on the original selected subset,which can create
a final-selection mismatch. Full recording risk must be evaluated separately.

## Scope

All source roles are exposed fitting/resubstitution. Three critic seeds are not three forecast producers.
Task:obs8/pred12 at raw stride12, detector-silver/local image coordinates.
Not metric,seconds-level,human gold,true3D,foundation or newly successful world dynamics.
No deployment change,Stage5C execution,SMC or submission-ready claim.
