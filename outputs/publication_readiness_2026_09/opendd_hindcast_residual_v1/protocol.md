# Hindcast-Anchored Forecast Training

**Question.** Does fitting the *emitted* forecast around the causal matched
hindcast, rather than fitting a direct forecast and blending it afterwards,
improve on simple motion? Does adding explicit positive-harm supervision
help without sacrificing utility? The preceding frozen cross-source test
lost to matched hindcast even on complete trajectories.

**Data.** Existing exposed OpenDD development cohort only: seven locations,
8 observed and 12 predicted samples. Leave one entire location out of each
fit and of its easy/hard cut estimation; three fixed seeds (17, 29, 43).
Fit locations contribute uniformly. Future positions, label availability
and easy membership are loss/readout inputs only. No TUMDOT or DroneCrowd
paths are opened by this experiment.

**Fixed arms.** Train all three from the same initial weights and draw order
within a location-seed pair. Each emits the 5% CV-to-hindcast blend plus a
bounded 5%-scaled, local heading velocity adjustment. Initialize the final
layer to zero, so the untrained network is exactly the matched hindcast.
The MLP and 600 updates are shared across arms. The sample-relative loss
uses causal past RMS displacement as scale:

1. `observed`: observed point errors only.
2. `completion`: add missing-step forecast disagreement against the exact
   matched hindcast, including fully unobserved paths.
3. `easy_guard`: add explicit positive ADE harm against CV on known easy
   rows and the maximum CV disagreement for fully unobserved rows. The
   fixed weight is 2. No test selection or safety threshold search.

Compare the original fixed 5% direct-model blend, 5% past-only hindcast,
and all three trained arms to CV and to hindcast. Fix intervention support
to the old legal menu and the nonidentity incumbent, before using labels.
All edited arms must match recipients and the common coefficient. They do
not necessarily match trajectory displacement. CV is an unedited control.
Record observed, complete-only and missing-step lower comparisons, ADE/FDE
and the unchanged 2% recording-level test, including undefined views.
Average seeds before a paired 3,000-draw location bootstrap (seven sites
from an exposed source, not independent confirmation).

The prespecified method contrast is `easy_guard` versus matched hindcast
and CV. A useful modeling result requires positive lower confidence bounds
against both, not merely a better comparison with the old neural forecast.
Keep the original recording safety readout separate; report undefined
risks as failures. Neither threshold choice nor deployment is authorized
by this exploratory experiment. No Stage5C or SMC.
