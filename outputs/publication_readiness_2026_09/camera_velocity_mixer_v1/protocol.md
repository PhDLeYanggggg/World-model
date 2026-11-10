# Velocity-only camera forecast experiment

This is a development-only test of the forecast before any joint selection.
The fixed 8-to-12 task remains in native image pixels. Use the seven existing
OpenDD donor cohorts, seed 17/29/43, 800 updates per seed and 128 examples
per batch. Compare clean fitting, identical synthetic-noise fitting, and
the same noise plus an easy-row CV harm penalty. Use matched draws across
arms. Select no checkpoint or threshold on the camera outcomes.

All forecasts remain anchored to the last detection and linear in time.
A small MLP learns convex weights on CV, OLS4, OLS8, median4 and median8
velocities using normalized observed steps and slope similarities. The
last arm adds twice the positive excess over 1.02 times CV error on the
lowest positive-CV-error donor quartile within each training batch. This
is a training regularizer, not calibration or a relaxation of the risk rule.

Freeze the nine new predictions before reading the already exposed WEPDTOF
Convenience Store pilot outcomes. Compare all controls, including both
pre-existing spline fits, on the same camera histories, masks and anchors.
Report observed ADE/FDE and the original two-reference and 2% risk metrics.
No external confirmation or non-pilot WEPDTOF label is read, no deployment
changes, and no choice is made from this single camera.
