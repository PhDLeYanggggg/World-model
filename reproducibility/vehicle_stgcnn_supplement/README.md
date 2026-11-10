# OpenDD: a matched Social-STGCNN comparison

Run `python3 -I -B -S verify.py` after extracting this folder.
The verifier reconstructs location-averaged estimates and paired
bootstrap intervals from the 21 anonymous location-seed receipts.
It keeps the v31 replay unchanged. This checks scalar aggregation,
not training or inference.

The fixed single-trajectory adaptation uses the author core from
Mohamed et al., Social-STGCNN (CVPR 2020), with the same
causal OpenDD task as the prior EqMotion comparison. It is not a
reproduction of the published pedestrian benchmark. The archive
contains no trajectories, learned weights, or third-party code.
