# Sources for the forecast replacement study

I checked the 17 existing entries against the sources below and added three
references where the source attribution or method distinction was missing.
MovingDroneCrowd++ now identifies arXiv v3 explicitly: the earlier v1 lists
five authors, while v3 lists the six in the manuscript. This specifies the
cited paper version, not a change in the locally evaluated annotations.
Songdo cites both the exact data release and the publication requested by its
creators. No source listed as a comparison has been newly rerun.

| Entry | Source checked | Use in the manuscript |
| --- | --- | --- |
| Scholler et al. | [TUM publication record](https://portal.fis.tum.de/en/publications/what-the-constant-velocity-model-can-teach-us-about-pedestrian-mo/) | Strong constant-velocity comparator, not a universal superiority claim. |
| Mao et al. | [PMLR](https://proceedings.mlr.press/v235/mao24d.html) | Cost-aware regression deferral. The forecast edits here share a query budget. |
| Fan et al. | [arXiv v3](https://arxiv.org/abs/2601.12500v3) and [v1](https://arxiv.org/abs/2601.12500v1) | MovingDroneCrowd++ provenance. The local forecast task is not the publisher's counting or tracking benchmark. |
| Xu et al. (EqMotion) | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_CVPR_2023_paper.html) | Fixed-head adaptation; original multimodal scores are not claimed. |
| Wen et al. | [CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Wen_Detection_Tracking_and_Counting_Meets_Drones_in_Crowds_A_Benchmark_CVPR_2021_paper.html) | DroneCrowd provenance. The confirmation subset is still sealed. |
| Elmachtoub and Grigas | [INFORMS](https://pubsonline.informs.org/doi/10.1287/mnsc.2020.3922) | Decision-focused training; retain the 2022 issue year rather than the 2021 online-first year. |
| Wilder et al. | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/3982) | Decision-focused training, not the solver used here. |
| Weng et al. | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Weng_Joint_Metrics_Matter_A_Better_Standard_for_Trajectory_Forecasting_ICCV_2023_paper.html) | Joint evaluation motivation. The relative-path check is not a joint best-of-modes score. |
| Breuer et al. | [IEEE](https://ieeexplore.ieee.org/abstract/document/9294301) and [dataset project](https://l3pilot.eu/data/opendd.html) | OpenDD provenance for the exposed development study. |
| Yu and Zhou | [Author manuscript](https://arxiv.org/abs/2108.08259) and [institutional record](https://pure.psu.edu/en/publications/towards-robust-human-trajectory-prediction-in-raw-videos/) | Re-tracking to improve past trajectories. |
| Li et al. | [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Li_NATRA_Noise-Agnostic_Framework_for_Trajectory_Prediction_with_Noisy_Observations_ICCV_2025_paper.html) | NATRA denoising of noisy past observations. |
| Fujii et al. | [Authors' publication page](https://research.nvidia.com/labs/twn/publication/icra_2026_realtraj/) | Detection-based forecasting; the ICRA 2026 version is distinct from the 2024 preprint date. |
| Da Silva Martins et al. | [arXiv](https://arxiv.org/abs/2608.30899) | Propagating tracking uncertainty into a forecast, not filling missing evaluation labels. |
| Tezcan et al. | [CVF proceedings](https://openaccess.thecvf.com/WACV2022) | WEPDTOF provenance and overhead fisheye input. |
| Xu et al. (loss-conditioned execution) | [arXiv](https://arxiv.org/abs/2609.15801) | A fixed proposal versus persistence; their expected-risk guarantee is not claimed here. |
| Wang | [arXiv](https://arxiv.org/abs/2609.14997) | AV mode pairing with preserved marginals; not the same as choosing recipients at a fixed replacement budget. |
| Songdo data release | [Zenodo v2](https://zenodo.org/records/17924857) | Exact release used; not a claim that the paper's horizon or coordinate metrics have been calibrated across datasets. |
| Xie and Alsinglawi | [arXiv](https://arxiv.org/abs/2610.08322) | Added: graph-based abstention of a whole multivariate forecast, rather than replacing a subset of agents' paths. |
| Angelopoulos et al. | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html) | Added: conformal expected-risk guarantees under assumptions. Adding an edit can raise or lower the observed harm ratio here, and the selected easy reference can be empty. |
| Fonod et al. | [Article DOI](https://doi.org/10.1016/j.trc.2025.105205) and [authors' record](https://arxiv.org/abs/2411.02136) | Added: published trajectory extraction article explicitly requested by the data release. |

## Local build

```sh
python scripts/build_m3w_verified_references_candidate.py --compiler /path/to/tectonic
```

The five-page PDF and source remain local. The build changes only the audited
reference text and its discussion, and checks the original title, abstract,
disclosure and experiment statements are preserved. [Build receipt](receipt.json).
The [v3 archive](../focused_review_supplement_v1/README.md) still replays the
original counts; the [Songdo menu audit](../songdo_frozen_menu_capacity_v1/README.md)
replays the additional finite-menu calculation. No label was opened and no
portal action was taken.
