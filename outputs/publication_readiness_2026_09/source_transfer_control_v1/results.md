# Matched Modality and Exposed-Source Results

Fresh mask-only continuation:6 branches,48000 new updates. RGB branches and all parents are cached_verified.
Fresh inference on6944 previously explored bookstore rows,7recordings,181 scopedagents,1physicalsite.
All18 fixed predictor states retained. No winner selection, main-role scoring or deployment.

| Arm | Endpoint | Policy | Gain vs CV (%) | Conditional recording95% CI | Seed range | Hard gain (%) | Zero-target harm (pixels) | Actual change rate |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| mask_only | parent_2k | uncontrolled | -2.078693 | [-4.839028,-1.262403] | [-2.290643,-1.969650] | -0.050949 | 0.02357670 | 100.000% |
| mask_only | parent_2k | fixed_probability_guard | -0.007154 | [-0.029264,-0.001974] | [-0.010266,-0.005495] | -0.001419 | 0.00005336 | 0.288% |
| mask_only | constant | uncontrolled | -4.232043 | [-11.409078,-2.341027] | [-4.561579,-3.782941] | -0.104209 | 0.05096471 | 100.000% |
| mask_only | constant | fixed_probability_guard | -0.031071 | [-0.145438,-0.006982] | [-0.041066,-0.022558] | -0.000615 | 0.00021128 | 0.288% |
| mask_only | cosine | uncontrolled | -1.743843 | [-5.899569,-0.810831] | [-2.072074,-1.469641] | -0.058039 | 0.02243060 | 100.000% |
| mask_only | cosine | fixed_probability_guard | -0.018661 | [-0.092316,-0.003244] | [-0.026362,-0.014017] | -0.000676 | 0.00013563 | 0.288% |
| past_rgb | parent_2k | uncontrolled | -2.414075 | [-5.583697,-1.460604] | [-3.131568,-1.810933] | -0.072511 | 0.02701009 | 100.000% |
| past_rgb | parent_2k | fixed_probability_guard | -0.051955 | [-0.161450,-0.024432] | [-0.053198,-0.050350] | -0.002459 | 0.00055752 | 2.021% |
| past_rgb | constant | uncontrolled | -5.762899 | [-15.942086,-3.140636] | [-8.202991,-4.503836] | -0.173316 | 0.06923900 | 100.000% |
| past_rgb | constant | fixed_probability_guard | -0.230984 | [-0.782708,-0.099891] | [-0.369213,-0.136983] | +0.001390 | 0.00225171 | 2.021% |
| past_rgb | cosine | uncontrolled | -2.080643 | [-6.984827,-0.989523] | [-2.580707,-1.774655] | -0.054830 | 0.02675066 | 100.000% |
| past_rgb | cosine | fixed_probability_guard | -0.115480 | [-0.440288,-0.043537] | [-0.155217,-0.075838] | +0.002099 | 0.00099810 | 2.021% |

## Paired Contrasts

Gain difference uses the same stationary CV denominator, in percentage points. Positive favors the named candidate.
Guarded RGB contrasts change both candidate and frozen classifier probabilities, not the image modality alone.

| Contrast | Policy | Gain difference (pp) | Conditional recording95% CI |
| --- | --- | ---: | --- |
| rgb_minus_mask:parent_2k | uncontrolled | -0.335382 | [-0.749840,-0.199082] |
| rgb_minus_mask:constant | uncontrolled | -1.530856 | [-4.540933,-0.789856] |
| rgb_minus_mask:cosine | uncontrolled | -0.336800 | [-1.085118,-0.166939] |
| endpoint_minus_parent:mask_only:constant | uncontrolled | -2.153350 | [-6.633778,-1.065658] |
| endpoint_minus_parent:mask_only:cosine | uncontrolled | +0.334850 | [-1.224490,+0.834010] |
| endpoint_minus_parent:past_rgb:constant | uncontrolled | -3.348824 | [-10.453298,-1.661607] |
| endpoint_minus_parent:past_rgb:cosine | uncontrolled | +0.333432 | [-1.560292,+0.886864] |
| rgb_minus_mask:parent_2k | fixed_probability_guard | -0.044801 | [-0.128969,-0.021756] |
| rgb_minus_mask:constant | fixed_probability_guard | -0.199913 | [-0.672268,-0.088057] |
| rgb_minus_mask:cosine | fixed_probability_guard | -0.096819 | [-0.371848,-0.038143] |
| endpoint_minus_parent:mask_only:constant | fixed_probability_guard | -0.023918 | [-0.111396,-0.004777] |
| endpoint_minus_parent:mask_only:cosine | fixed_probability_guard | -0.011508 | [-0.060608,-0.001017] |
| endpoint_minus_parent:past_rgb:constant | fixed_probability_guard | -0.179030 | [-0.631625,-0.071393] |
| endpoint_minus_parent:past_rgb:cosine | fixed_probability_guard | -0.063526 | [-0.296799,-0.009498] |

## Evidence Limits

These2000 paired recording-block draws preserve window weighting and first average seed errors, not predictions. Recordings share one physical site; this is not a physical-scene generalization interval. Main confirmation is not opened.
Easy CV error is zero, so its percentage degradation is undefined. Absolute harm must not be converted into an invented passing percentage. Binary future oracles are diagnostics, never deployable policies.
Eight supplied past annotation points to twelve stride12source steps (+144rawframes); no seconds or cross-dataset time equivalence. Source normalization and native annotation pixels are not metric coordinates. Offline/interpolated labels are not human intention gold.
Verification:42 exact replays,3four-way matched sampler streams,all finite bounded forecasts,unsupported contexts exactly zero,202 artifacts unchanged on completed resume and repeated evaluation,zero new updates.
