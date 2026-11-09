# Same-budget direct forecasting on the higher-coverage OpenDD roster

The previous direct spline improved observed steps on exposed moving
drone development data but could not beat OLS4 conservatively. Here I
test whether the spline improves the stronger CV reference on the
existing exposed OpenDD pedestrian roster, where most future paths
are complete. This is an additional corpus, not independent
confirmation or a controlled intervention on missingness.

The registered OpenDD roster has seven held locations, 202 recordings,
10,475 windows, eight observations and twelve requested steps at
source stride 12. Use each held location exactly once per seed; fit
on the other six with three seeds. Reuse the prior ego-only EqMotion
training budget: 800 updates, 32 site-balanced observed-label rows
per update, identical RNG and sampling order, AdamW, and the same
robust loss using observed motion scale. EqMotion is a fixed-head
source-verified local adaptation; no pretrained weights or published
best-of-multiple-forecasts results are compared.

The spline controls use the same architecture and no peer input:
OLS4 anchor with centered history, CV anchor with the same history
representation, and CV anchor with history first transformed into
the focal constant-velocity frame. The last transform also changes
the feature scale through the same observation-only preprocessing.
No extra calibration or architecture selection occurs on the held
locations. Preserve all direct forecasts before scoring.

The primary comparison is comoving spline against CV. Also compare
OLS4, frozen centered-peer and comoving EqMotion paths, and the
two matched spline controls. Report full-roster observed and
missing-step-conservative gain, complete ADE, available endpoint FDE,
the original per-recording 2% risk, and 3,000 paired location
bootstrap draws after averaging seeds. Require the primary lower
interval to be positive before treating a spline as a strong direct
candidate. No policy changes, no deployment, no held threshold search
and no DroneCrowd calibration or confirmation access.
