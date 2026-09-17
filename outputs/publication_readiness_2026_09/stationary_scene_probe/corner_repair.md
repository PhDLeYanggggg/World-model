# Shared-Corner Geometry Repair

The first static-scene experiment completed all 72 models and saved all outcomes.
Its geometry logic marked 35 Hotel rows' reference frame undefined. Inspection
identified that two adjacent obstacle segments may return the same closest corner
and therefore the same normal. Equal distance to those duplicate surface points
is not directional ambiguity. The original implementation conflated primitive
ties with distinct closest-surface ties.

Repair: equal-distance surface points are ambiguous only when their coordinates
differ at the declared relative tolerance. A query on a surface or at a circle
center remains directionless. Opposite equal-distance walls remain ambiguous.
An explicit shared-corner regression check and the existing transformation/future
input tests cover the correction.

Preserve initial source, config, checkpoints and metrics in stationary_scene_probe;
archive the exact initial sources under its local source_snapshot. Run the corrected
implementation with a new registration and stationary_scene_probe_v2 output.
Do not alter the old registration hashes to disguise this change as an exact replay.
This repair changes geometric frame support/features and their derived regression
labels, not the underlying future coordinates, fit/held rows, primary metric,
eight/twelve-step task, model settings, seeds or fixed probability gate.

All original unrestricted regressors were worse than CV; scene features produced
some start-probability lift, not trajectory success. Keep those results visible.
The repair does not authorize selecting thresholds or models from held outcomes,
using destinations/groups, admitting reference image pixels, or claiming physical
geometry. The corrected run must report every setting, positive or negative.
