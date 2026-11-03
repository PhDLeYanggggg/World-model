# What New Moving-Camera Inputs Would Need To Prove

## Primary Work Already Excludes Simple Novelty Claims

[Neumann and Vedaldi, CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Neumann_Pedestrian_and_Ego-Vehicle_Trajectory_Prediction_From_Monocular_Camera_CVPR_2021_paper.html)
already separate pedestrian motion from ego-motion using learned pose/view
normalization. The retrieved primary method excerpt describes prediction in
normalized2Dviews. Camera-motion compensation, view normalization and combining
visual and trajectory streams therefore cannot be claimed as new M3W ideas.
Read scope this turn: official abstract and indexed primary method/Figure3
excerpt; direct full-PDF retrieval was403, not silently counted as a full read.

[Bhattacharyya etal., CVPR2018](https://arxiv.org/abs/1711.09026)
already jointly model vehicle motion and person trajectories with uncertainty.
The official primary excerpt distinguishes observation and model uncertainty.
Predictive uncertainty or joint ego/person forecasting alone is not our novelty.
Read scope this turn: author arXiv abstract and indexed CVF introduction excerpt;
direct PDF retrieval timed out. No published numerical score is used here.

## Prospective M3W Question, Not An Established Contribution

Does past-only visual/shared-view context contain additional information about
**relative benefit and harm of neural intervention**, especially on the joint
decisions selected by a policy, beyond histories and frozen interaction features?
This is not answered by adding a camera encoder or reducing average forecast
loss. SOURCE already shows strong forecasts can coexist with failed selected
costs, and fourfold training does not repair its own selected easy harm.

Before new fitting, finish source exclusions and producer/role admission.
Any later view-motion input must use current/past imagery only. Middle/last
images in this audit cannot be reused as future predictive features. The
image-to-image transforms here are source diagnostics, not calibrated camera
poses, metric ground maps or learned world dynamics.

The minimum later experiment would freeze the predictor and compare no-control,
independent-agent, uniform-scene and joint decisions with matched intervention
counts, plus the complete operational2%risk checks. Equal-count controls test
utility, not safety; empty controls remain not estimable. Require three actual
seeds, grouped uncertainty, and separate past-visual/shared-context ablations.
Neither uncertainty/error correlation nor a positive known-label mean certifies
the complete missing-label risk contract. No test thresholds or source-role
changes may be chosen from predictive outcomes.

This is an evidence-driven hypothesis for the next admitted experiment, not a
new method already implemented, a literature-completeness certificate, or a
reason to open protected outcomes. Original main hypothesis/gates remain
unmet; no submission, deployment,Stage5C orSMC is promoted.
