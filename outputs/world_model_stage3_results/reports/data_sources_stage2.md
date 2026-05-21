# Stage 2 Real Data Sources

This file records candidate long-trajectory data sources for moving beyond the short AerialMPT bauma3 slice.

| Dataset | Downloadable | Trajectories | Frames | Ground-plane coordinates | Homography / scene map | t+100 evaluation | Current decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Stanford Drone Dataset | Yes | Yes | Yes | Pixel annotations with scene context; can be mapped if calibrated | Scene images; homography must be supplied/estimated | Likely, depending on frame rate and track length | Best next real-data target |
| TrajNet++ | Yes | Yes | Usually trajectory tables, not raw images | Trajectory coordinates | Homography varies by source | Often yes | Good for trajectory-only training/eval |
| ETH/UCY | Yes | Yes | Fixed-camera pedestrian scenes | World/pixel annotations depending on version | Homographies often available in common releases | Often yes after horizon alignment | Good compact benchmark |
| AerialMPT | Partially in current project | Yes for selected slice, but bauma3 slice is short | Yes | Weak homography currently used | No real calibration in current scaffold | No for selected bauma3 16-frame slice | Keep only t+12 verified, t+100 qualitative |

## Source Notes

- Stanford Drone Dataset official page: https://cvgl.stanford.edu/projects/uav_data/ . It describes eight unique scenes and multiple agent types including pedestrians.
- TrajNet++ official EPFL/VITA dataset page: https://www.epfl.ch/labs/vita/datasets/ . It describes TrajNet++ as an interaction-centric human trajectory forecasting benchmark.
- ETH BIWI Walking Pedestrians official ETH dataset page: https://vision.ee.ethz.ch/datsets.html . It lists manually annotated walking pedestrians in busy bird-eye scenarios with annotations/videos.
- UCY crowd data official URL is commonly cited as https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data . The page was not reachable during this Stage 2 run, so it is not yet wired into the loader.

The current Stage 2 demo does not automatically download these datasets. Next step is to add one real long-trajectory loader and a calibrated scene file.
