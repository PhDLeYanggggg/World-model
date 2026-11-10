# OpenDD vehicles: coordinate directions without reassigning edits

The sparse pedestrian controller has ten double-edit query-seed views,
nine from one location. Use the established vehicle two-edit development
comparison instead. Reuse both additive-selected recipients in all
8,874 views from 493 recordings. Rotate their already frozen residuals
relative to CV without changing the residual norm at any step.

Enumerate all 169 pairs over 13 fixed angles, including zero. Using
labels only in this oracle, check the original risk and CV/OLS4
conditions, no conservative trajectory loss to additive, and a
positive full-grid relative-motion bound versus additive. Report
the same-angle and single-rotation restrictions separately.

The causal comparison forces both recipients to rotate by a nonzero
angle so the edit and rotation counts match. The independent arm
minimizes distance to OLS4 per actor, the shared arm restricts the
two angles to be equal, and the coupled arm adds distance to OLS4's
relative trajectory with fixed weight one. None sees future labels.
Report original risk, conservative trajectory and relative-motion
contrasts, including direct coupled versus independent, with
3,000 location draws after averaging calibration directions and
seeds. No threshold change, new training, external outcomes or release.
