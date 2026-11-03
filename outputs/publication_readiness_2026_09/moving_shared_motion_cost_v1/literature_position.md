# Shared Motion Is A Testable Cue, Not A Novelty Claim

Primary sources checked3November2026 while the forecast controls still fit.

[Neumann and Vedaldi,CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Neumann_Pedestrian_and_Ego-Vehicle_Trajectory_Prediction_From_Monocular_Camera_CVPR_2021_paper.html)
already separate pedestrian movement from camera movement using learned future
pose and normalized views. Our trajectory-only crowd summaries neither estimate
camera pose nor reconstruct a verified ground coordinate system. No novelty
claim for noticing camera motion, and no matched score comparison is made.

[Yin etal.,IJCAI2021](https://ijcai.org/proceedings/2021/174)
already fuse pedestrian trajectories, ego-vehicle speed and optical flow with a
multimodal Transformer. Our48fields contain neither speed sensors nor optical
flow. Richer temporal/social input alone is not the proposed paper contribution.

[The author's MovingDroneCrowd repository](https://github.com/fyw1999/MovingDroneCrowd)
distinguishes the ICCV2025conference release from the extended
MovingDroneCrowd++counting/tracking benchmark. The local source snapshot,
source groups and derived native8/12forecasting protocol are version-bound;
they are not the author's published trajectory benchmark or performance claim.

The narrower hypothesis is whether new causal information improves the decision
to replace a strong motion floor, under baseline-relative positive-harm/easy
constraints and disjoint learned producer roles. Existing deferral and conformal
risk work already cover key ingredients; see the
[cost-decision prior-work note](../cost_decision_related_work_20261029.md).
A significant feature effect would support information value, not by itself a
new calibrated joint-decision method. A failed full risk gate stays failed even
if average forecast utility improves. No contribution is declared before data.
