# Visual Forecasting Probe: Prior Art and Claim Boundary

Checked against primary sources on 2026-09-17 while the registered fits ran.
This note does not change their models, training budget or analysis.

## What Is Already Known

Uhlemann, Fent and Lienkamp examine single-trajectory ETH/UCY prediction, history
requirements and runtime. Their discussion identifies false movement in static
cases, difficulty with state changes, and suggests combining state-change
recognition with a constant-velocity predictor. Our stationary-failure diagnosis
and the phrase "neural model plus CV" are therefore not, by themselves, new
contributions. Their results also do not prove that our own modified split,
coordinates or normalization yield the same conclusions.
[T-ITS paper, Sections III--V](https://arxiv.org/html/2308.05194v3).

Y-Net models goal and path uncertainty in a scene-aware forecasting framework.
Consequently, adding a scene image or a goal component is not a novelty claim.
Our current experiment is a deterministic local-appearance information probe,
not a Y-Net reproduction and not a best-of-many trajectory comparison.
[Original ICCV 2021 paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Mangalam_From_Goals_Waypoints__Paths_to_Long_Term_Human_Trajectory_ICCV_2021_paper.pdf).

Bahari and colleagues certify trajectory-predictor robustness to bounded input
perturbations. This is a different question from whether a selected predictor
has lower realized forecast error than a causal baseline on an unseen scene.
Neither their certification nor our empirical easy-error ceiling should be
described as a physical collision-safety certificate.
[Original CVPR 2025 paper](https://arxiv.org/html/2403.13778v2).

## The Distinction Still To Establish

Our proposed contribution must be an evidenced decision mechanism: learn excess
gain and harm relative to a strong baseline, coordinate interventions within a
scene, and quantify the conditions under which selective prediction helps. The
visual study tests a prerequisite, useful predictive information, not that entire
claim. A mask-only control separates pixel content from coverage cues. Joint
versus independent selection still needs matched predictors and intervention
budgets; scene-level risk control still needs adequate independent calibration
support. Three historically exposed fit scenes do not supply that support.

A null visual result cannot show that visual prediction is impossible. The tested
small CNN, fixed local crop, offline annotation clock and normalized target are
specific choices. Potential repairs, such as orientation-aware image features or
pretrained visual representations, need new registered comparisons, not a new
name for the same negative run. No results from the ongoing fits were used to
select a replacement model or reinterpret the primary metric in this note.
