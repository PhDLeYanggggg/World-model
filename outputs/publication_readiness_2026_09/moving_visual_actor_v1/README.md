# Does Past-Image Context Improve The Trajectory Actor?

I moved the same verified past-image measurements into the trajectory
predictor, rather than only the intervention controller. This experiment
changes the actual neural actor and retains the original whole12loss.

Component progression screen: **FAIL**.
Primary motion-minus-quality known difference: **-1.8039pp**.
Paired completion lower: **-10.9309pp**, exploratory95%
source-bootstrap interval **[-17.8261, -5.2878]**.
Known-positive sources:7/14. Fold means:{'0': -6.449304420538456, '1': 1.6802380326513584}.

**This is not a deployment result or independent confirmation.** No new
controller is trained. The original2%/support/NA requirements remain.

## Full Paired Prefix Matrix

| Contrast | Prefix | Control gain vs CV % | Treatment gain vs CV % | Paired known pp | Paired lower pp |
|---|---:|---:|---:|---:|---:|
| primary | 1 | -8.8041 | -11.3398 | -2.5357 | -7.1653 |
| primary | 3 | +7.7175 | +6.3007 | -1.4167 | -5.9701 |
| primary | 6 | +13.8902 | +12.1306 | -1.7596 | -8.4209 |
| primary | 12 | +16.1152 | +14.3113 | -1.8039 | -10.9309 |
| quality_bundle | 1 | -13.6826 | -8.8041 | +4.8784 | -0.4959 |
| quality_bundle | 3 | +6.6831 | +7.7175 | +1.0344 | -4.4255 |
| quality_bundle | 6 | +13.6088 | +13.8902 | +0.2814 | -7.1509 |
| quality_bundle | 12 | +15.3777 | +16.1152 | +0.7374 | -12.3629 |
| motion_bundle | 1 | -13.6826 | -11.3398 | +2.3428 | -2.8124 |
| motion_bundle | 3 | +6.6831 | +6.3007 | -0.3823 | -6.3373 |
| motion_bundle | 6 | +13.6088 | +12.1306 | -1.4782 | -9.5320 |
| motion_bundle | 12 | +15.3777 | +14.3113 | -1.0664 | -15.0206 |

Primary is full motion versus the matched quality/identity-motion control.
The two raw comparisons are bundle contrasts, not pure motion attribution.
Weights balance queries within sources, then average sources. The lower
statistic bounds missing-label uncertainty; it is not the actual unknown
forecast error. Prefix-only improvements cannot replace a failed whole12.

## Every Source

| Source | Contrast | Rows | Known | Partial | Unknown | Control gain % | New gain % | Paired pp | Paired lower pp | New easy +harm % | New signed easy loss % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| moving_source_component_048 | primary | 2038 | 1997 | 618 | 41 | +39.8633 | +36.9416 | -2.9216 | -6.8394 | +33.8901 | +23.3372 |
| moving_source_component_067 | primary | 4159 | 4006 | 1538 | 153 | +45.5715 | +18.6336 | -26.9379 | -45.3762 | +85.9533 | +76.2043 |
| moving_source_component_048 | quality_bundle | 2038 | 1997 | 618 | 41 | +32.8183 | +39.8633 | +7.0450 | -4.5567 | +44.3877 | +31.2069 |
| moving_source_component_067 | quality_bundle | 4159 | 4006 | 1538 | 153 | +39.2407 | +45.5715 | +6.3308 | -5.9235 | +56.7761 | +41.2376 |
| moving_source_component_048 | motion_bundle | 2038 | 1997 | 618 | 41 | +32.8183 | +36.9416 | +4.1233 | -10.3693 | +33.8901 | +23.3372 |
| moving_source_component_067 | motion_bundle | 4159 | 4006 | 1538 | 153 | +39.2407 | +18.6336 | -20.6071 | -45.6145 | +85.9533 | +76.2043 |
| moving_source_component_047 | primary | 2198 | 2167 | 450 | 31 | -13.6161 | -14.2542 | -0.6381 | -2.3980 | +54.9967 | +32.9195 |
| moving_source_component_052 | primary | 536 | 518 | 73 | 18 | +62.6574 | +60.2807 | -2.3767 | -11.4914 | +42.1935 | +26.3919 |
| moving_source_component_047 | quality_bundle | 2198 | 2167 | 450 | 31 | -11.6184 | -13.6161 | -1.9977 | -3.6296 | +49.0066 | +26.2326 |
| moving_source_component_052 | quality_bundle | 536 | 518 | 73 | 18 | +61.3592 | +62.6574 | +1.2983 | -10.4238 | +45.8827 | +34.1501 |
| moving_source_component_047 | motion_bundle | 2198 | 2167 | 450 | 31 | -11.6184 | -14.2542 | -2.6358 | -4.6317 | +54.9967 | +32.9195 |
| moving_source_component_052 | motion_bundle | 536 | 518 | 73 | 18 | +61.3592 | +60.2807 | -1.0784 | -16.2348 | +42.1935 | +26.3919 |
| moving_source_component_051 | primary | 7196 | 7128 | 874 | 68 | +24.4186 | +18.6361 | -5.7825 | -12.2496 | +96.5449 | +87.0524 |
| moving_source_component_062 | primary | 37829 | 37472 | 4129 | 357 | -35.6134 | -35.6524 | -0.0390 | -1.9154 | +218.8655 | +215.4215 |
| moving_source_component_051 | quality_bundle | 7196 | 7128 | 874 | 68 | +5.4075 | +24.4186 | +19.0111 | +12.0226 | +69.9692 | +55.3038 |
| moving_source_component_062 | quality_bundle | 37829 | 37472 | 4129 | 357 | -20.1659 | -35.6134 | -15.4475 | -17.5709 | +217.7383 | +213.9351 |
| moving_source_component_051 | motion_bundle | 7196 | 7128 | 874 | 68 | +5.4075 | +18.6361 | +13.2286 | +6.5763 | +96.5449 | +87.0524 |
| moving_source_component_062 | motion_bundle | 37829 | 37472 | 4129 | 357 | -20.1659 | -35.6524 | -15.4865 | -18.0572 | +218.8655 | +215.4215 |
| moving_source_component_053 | primary | 1137 | 1095 | 570 | 42 | +29.3331 | +34.4558 | +5.1228 | -5.4049 | +238.3413 | +232.1520 |
| moving_source_component_065 | primary | 1042 | 941 | 853 | 101 | +17.8417 | +20.5940 | +2.7523 | -20.6784 | +41.6719 | +18.5311 |
| moving_source_component_076 | primary | 3342 | 3184 | 1604 | 158 | -16.7264 | -15.3259 | +1.4005 | -5.4454 | +135.9870 | +129.5578 |
| moving_source_component_053 | quality_bundle | 1137 | 1095 | 570 | 42 | +30.0694 | +29.3331 | -0.7363 | -21.9071 | +226.0887 | +218.4766 |
| moving_source_component_065 | quality_bundle | 1042 | 941 | 853 | 101 | +14.6994 | +17.8417 | +3.1423 | -60.9796 | +41.9333 | +21.7520 |
| moving_source_component_076 | quality_bundle | 3342 | 3184 | 1604 | 158 | -12.4842 | -16.7264 | -4.2422 | -14.7358 | +137.9945 | +132.2441 |
| moving_source_component_053 | motion_bundle | 1137 | 1095 | 570 | 42 | +30.0694 | +34.4558 | +4.3864 | -7.7191 | +238.3413 | +232.1520 |
| moving_source_component_065 | motion_bundle | 1042 | 941 | 853 | 101 | +14.6994 | +20.5940 | +5.8946 | -40.8482 | +41.6719 | +18.5311 |
| moving_source_component_076 | motion_bundle | 3342 | 3184 | 1604 | 158 | -12.4842 | -15.3259 | -2.8417 | -11.5523 | +135.9870 | +129.5578 |
| moving_source_component_080 | primary | 10670 | 10523 | 1806 | 147 | +8.8809 | +13.1510 | +4.2701 | +2.4066 | +47.6483 | +36.4889 |
| moving_source_component_082 | primary | 6225 | 5930 | 3572 | 295 | -12.2150 | -7.3311 | +4.8838 | +0.5049 | +94.9105 | +86.3937 |
| moving_source_component_087 | primary | 2847 | 2653 | 1778 | 194 | +15.1484 | +19.5059 | +4.3575 | -24.5450 | +164.2862 | +147.6035 |
| moving_source_component_080 | quality_bundle | 10670 | 10523 | 1806 | 147 | +14.4769 | +8.8809 | -5.5960 | -11.2391 | +51.8515 | +42.6709 |
| moving_source_component_082 | quality_bundle | 6225 | 5930 | 3572 | 295 | -15.4158 | -12.2150 | +3.2009 | -3.0455 | +104.7300 | +95.1156 |
| moving_source_component_087 | quality_bundle | 2847 | 2653 | 1778 | 194 | +17.6411 | +15.1484 | -2.4926 | -17.9860 | +167.5941 | +150.6962 |
| moving_source_component_080 | motion_bundle | 10670 | 10523 | 1806 | 147 | +14.4769 | +13.1510 | -1.3258 | -5.4853 | +47.6483 | +36.4889 |
| moving_source_component_082 | motion_bundle | 6225 | 5930 | 3572 | 295 | -15.4158 | -7.3311 | +8.0847 | -1.0341 | +94.9105 | +86.3937 |
| moving_source_component_087 | motion_bundle | 2847 | 2653 | 1778 | 194 | +17.6411 | +19.5059 | +1.8649 | -33.8154 | +164.2862 | +147.6035 |
| moving_source_component_077 | primary | 6127 | 5943 | 2111 | 184 | +17.5222 | +17.6886 | +0.1664 | -3.7452 | +75.6352 | +63.1714 |
| moving_source_component_078 | primary | 40194 | 39159 | 11775 | 1035 | +42.5463 | +33.0348 | -9.5115 | -15.8546 | +22.6928 | +4.6795 |
| moving_source_component_077 | quality_bundle | 6127 | 5943 | 2111 | 184 | +23.5650 | +17.5222 | -6.0428 | -12.5807 | +107.8471 | +96.5792 |
| moving_source_component_078 | quality_bundle | 40194 | 39159 | 11775 | 1035 | +35.6952 | +42.5463 | +6.8511 | -0.5256 | +24.4700 | -0.0799 |
| moving_source_component_077 | motion_bundle | 6127 | 5943 | 2111 | 184 | +23.5650 | +17.6886 | -5.8764 | -12.2700 | +75.6352 | +63.1714 |
| moving_source_component_078 | motion_bundle | 40194 | 39159 | 11775 | 1035 | +35.6952 | +33.0348 | -2.6605 | -9.2325 | +22.6928 | +4.6795 |

Raw motion actor known-easy +harm exceeds2%in33/34recordings.
This is a raw actor diagnostic, not the full selected-policy certificate.
Undefined references remain NA. Positive-harm ratios can exceed100%;
they are not probabilities,physical safety or an identity certification.

## What Actually Ran

Fresh_run:twelve full4000update image-conditioned actors,one full4000update
raw-control refit and two full4000update treatment refits on the first
fixed source design. Do not call these twelve complete exact refits.
The real100+100versus200recovery pilot is exact;its200prefix is resumed
into its own full motion fit. A separate direct200pilot is additional.
Total actual optimizer updates:60,351including recovery,full refits and discarded/repeated updates.
Disk-guard recovery repeats151updates from the last atomic checkpoint;the scientific endpoint stays4000.
Each treatment has3,028,996nominal/582,404trainable parameters;unused
author heads remain frozen. The image branch adds1,728parameters.
Raw shared core/optimizer/sampler/RNG/common trace exact:True.

Cached_verified:six original raw OOF predictors and their125540donor
forecasts,the source split and past-image measurements. Fresh preparation
joins46causal action-invariant features to every original global ID.
Normalizers use past fitting rows only;no donor or own-B statistics.

The generic inference loader opens and hashes target files for integrity;
only causal inputs reach the model. Freeze-before-readout refers to
outcome scoring,not a target-unopened filesystem boundary.See execution_scope.md.

Not_run:new risk head fitting,new own-B/C/H/reserved readout,three forecaster
seeds,independent confirmation or a complete new deployment risk check.
All models use native arm64 CPU4/inter-op1/workers0. Final4000checkpoint,
no early stopping,threshold tuning,quick downgrade or cohort deletion.

## Scope

Native MovingDroneCrowd++ observation8/prediction12,stride1,image
coordinates and available-label ADE. Not SDDt50,meters,seconds,true3D,
foundation,human-gold or physical safety. The adapted image-conditioned
model does not automatically inherit the original core equivariance proof.
Bounding a latent offset does not bound trajectory error. The branch is
conventional conditioning,not a novel world-model method by itself.
The current paper is not submission-ready. Stage5C execution/SMC remain off.

[Protocol](protocol.md) | [Model scope](model_scope.md) | [Loss trace](loss_trace.csv) |
[Recording diagnostics](recording_diagnostics.csv) | [Operation](operation_zh.md)
