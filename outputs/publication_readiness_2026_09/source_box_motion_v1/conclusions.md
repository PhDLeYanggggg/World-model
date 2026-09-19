# Observed Box-Motion Comparison

## Scope

Result source: fresh_run for flow extraction,24 new Torch heads and readout;
cached_verified for original images and24 preceding conditioned references.
All15,430 source windows,29 recordings,four explored sites,three seeds retained.
Eight supplied-annotation observations ->12 predictions at12-raw-frame stride.
No main/bookstore/external scoring; no selection or deployment.

## Results

| Arm/source | Equal-site ADE gain vs CV | Conditional four-site95%CI | Static harm, annotation px | Nonzero-target gain | Binary CV/candidate oracle |
| --- | ---: | --- | ---: | ---: | ---: |
| geometry (cached_verified) | -0.00025066% | [-0.00063107, -0.00002717] | 0.00000644 | -0.00005468% | 0.00003027% |
| centered (cached_verified) | -0.00134153% | [-0.00365502, -0.00004997] | 0.00004117 | -0.00006461% | 0.00033943% |
| quality (fresh_run) | -0.00053454% | [-0.00102980, -0.00016415] | 0.00001046 | -0.00007624% | 0.00001514% |
| motion (fresh_run) | -0.00084041% | [-0.00118360, -0.00030231] | 0.00001485 | -0.00009867% | 0.00001997% |

Motion-minus-quality: -0.00030587 percentage points;
conditional95%CI [-0.000625063487561447, -3.2069741186998204e-05].
Positive held fits: quality 0/12;
motion 0/12.
Verdict: **no_positive_source_motion_forecast_evidence**.

The exact same63,960-parameter head is trained on quality-only versus
quality+motion tokens. Fixed importance-corrected uniform-row ADE, training-only
normalization/readout gain, same sampler streams and240,000updates.
No learned visual encoder or end-to-end pixel training is claimed.

## Measurement

23,890 unique observed pairs supply108,010 overlapping pair uses. Box support
98.874%; surround support
99.958%.
Rounded crop translation and per-axis video resize are restored before regional
flow pooling. The surrounding region is a background proxy, not verified camera
motion. Flow magnitude and support do not establish intent or forecasting value.
Annotation-box regions are not segmented bodies. All pair uses contain at least
one generated annotation; 8.514% contain an occlusion flag.
These are offline supplied annotations, not strict sensor-as-of observations.

51 zero-radius unsupported rows initially stopped the pilot before any update.
The pre-fit amendment retains them, zeros only unavailable motion channels and
preserves the original zero-forecast support rule. No retrospective row deletion.

## Compute And Verification

Nativearm64 CPU4/inter-op1/workers0. Fresh fitting 656.347s including the
100-update resume pilot; flow extraction 9.750s excluding
source-loading/hash verification.24 exact model replays and23,890 exact flow
pair replays.24 regenerated sampler streams match controls;12 arm pairs match.
Six OOF archives recomputed.32 future-target poison queries unchanged;24 illegal
training-role checks rejected. Completed resume adds0updates and preserves
144 artifacts. Checkpoints/caches remain local.

## Limits

The2,000 bootstrap resamples use four explored sites and shared training folds;
they are conditional uncertainty, not independent confirmation. Seeds are
averaged at the error level, not ensembled forecasts. Overlapping windows are
not independent. Static percentage degradation is undefined because CV error
is zero; absolute harm is reported, not converted to a2%pass.

No metric/seconds/true3D/foundation claim. Raw-frame t+50 is not rerun here.
Historical external selector gains remain exploratory. Stage5C and SMC are off.
The main research goal is active and unmet.
