# Fixed Source-Site Deferral Readout

## Material Passport

Fresh inference from all six frozen neural endpoints. Three dense controls are cached_verified and freshly replayed.
6944 queries, seven recordings, 181 scoped agents, one previously explored physical site. No training or threshold selection.
Seed errors are averaged, not forecasts. All intervals are conditional recording-block diagnostics, not independent scene confirmation.

| Variant | Output | Gain vs CV (%) | Conditional recording 95% CI | Hard gain (%) | Zero-target harm (pixels) | Actual intervention |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| expected_cost | proposal | -1.805004 | [-6.079203, -0.842182] | -0.057535 | 0.02309702 | 100.000% |
| expected_cost | hard_action | +0.000000 | [+0.000000, +0.000000] | +0.000000 | 0.00000000 | 0.000% |
| cost_supervised | proposal | -1.747853 | [-5.838922, -0.821269] | -0.050798 | 0.02245889 | 100.000% |
| cost_supervised | hard_action | -1.245558 | [-4.430101, -0.566922] | -0.036330 | 0.01581748 | 62.985% |
| dense_control | proposal | -1.743843 | [-5.899569, -0.810831] | -0.058039 | 0.02243060 | 100.000% |

## Paired Aggregate Contrasts

| Candidate | Reference | Difference (pp) | Conditional recording 95% CI |
| --- | --- | ---: | --- |
| expected_cost:proposal | dense_control:proposal | -0.061161 | [-0.195139, -0.021809] |
| expected_cost:hard_action | dense_control:proposal | +1.743843 | [+0.810831, +5.899569] |
| expected_cost:hard_action | expected_cost:proposal | +1.805004 | [+0.842182, +6.079203] |
| cost_supervised:proposal | dense_control:proposal | -0.004010 | [-0.060179, +0.083615] |
| cost_supervised:hard_action | dense_control:proposal | +0.498285 | [+0.242032, +1.438712] |
| cost_supervised:hard_action | cost_supervised:proposal | +0.502295 | [+0.247705, +1.385193] |

## Limits

Easy relative degradation is undefined at zero CV error; absolute harm is not a passed 2% gate.
All-baseline output and improvement over a bad neural control do not establish positive gain over the baseline.
Offline supplied annotations, stride12/+144 raw frames, annotation pixels/past-normalized coordinates only.
No meters/seconds, human intention gold, true-3D, foundation, deployment or submission-readiness claim. Stage5C and SMC remain off.
