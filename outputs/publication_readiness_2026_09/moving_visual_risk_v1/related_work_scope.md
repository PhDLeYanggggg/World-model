# What This Image Measurement Does Not Claim

The background-motion branch is an engineering test of a new observation for
the existing relative-risk controller. It is not a novel optical-flow method,
an odometry estimator or a calibrated world-coordinate reconstruction.

The measurement uses established sparse Lucas-Kanade tracking and robust
partial-affine fitting. The official API documentation describes the
[tracking interface](https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html)
and [affine estimator](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html).
These references establish the available algorithms, not their validity on
this drone dataset. The installed OpenCV4.14.0 signatures were checked locally;
the cited online API pages are4.13.0. Actual quality failures stay in the cohort.

Separating apparent pedestrian motion from moving-camera effects is prior art.
The publisher abstract for Neumann and Vedaldi's
[CVPR2021 paper](https://openaccess.thecvf.com/content/CVPR2021/html/Neumann_Pedestrian_and_Ego-Vehicle_Trajectory_Prediction_From_Monocular_Camera_CVPR_2021_paper.html)
describes ego-motion prediction and viewpoint normalization for monocular
on-board pedestrian forecasting. This is a different camera setting, not a
direct benchmark score or a claim of equivalence to this background proxy.
The abstract and bibliographic record were retrieved; direct full-paper access
returned403 in this session. No full-text comparison is claimed here.

The present hypothesis is narrower: can strictly past image information improve
held-source prediction of intervention gain, harm and label support sufficiently
to improve a joint policy under its unchanged recording-level contract? The
four matched arms distinguish native-image bounds, measurement quality and
motion. Standard motion compensation alone cannot establish method novelty.
Even a positive exploratory component contrast would still require independent
confirmation, strong matched controls and a defensible new decision-method
contribution before a paper-level claim.
