# Viewpoint Conditioning Is A Conventional Repair, Not A New Contribution

Neumann and Vedaldi already separate pedestrian/ego motion,normalize the view
and predict the camera viewpoint in their monocular forecasting method:
[CVPR2021 paper](https://openaccess.thecvf.com/content/CVPR2021/html/Neumann_Pedestrian_and_Ego-Vehicle_Trajectory_Prediction_From_Monocular_Camera_CVPR_2021_paper.html).
The official abstract and cached figure/view-normalization excerpt were read
this turn;direct full-PDF retrieval returned403. Do not claim a full method
reproduction from those excerpts. The proposed experiment instead uses an
existing2Daffine proxy and fixed last-affine extrapolation,not their learned
self-supervised future-pose estimator. Camera compensation itself is not new.

The EqMotion author core,MIT license and pinned source commit are inherited
unchanged from the source-control registration. This is deterministic K1,
not the published best-of20 protocol or a reported ETH/UCY result. The new
time-dependent image-coordinate preprocessor does not automatically inherit
the original model's equivariance theorem. A query-view transform is not a
homography to a calibrated ground plane,verified ego pose or true3D geometry.

The current project needs trustworthy baseline-relative neural intervention,
not another architecture name. This experiment can supply a candidate-quality
component result but cannot establish the original2%selected-policy contract,
independent confirmation or paper readiness. A good simple camera-CV baseline
without additional neural lift would support the geometric control,not a new
neural contribution. All positive and negative controls remain in the report.
