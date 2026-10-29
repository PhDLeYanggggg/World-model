# Forecast-Substitution Evidence, Revision 7

This anonymous English working draft adds the completed shared-reference and
static scene-context cost ablations to revision6. The old revision and its
four-study evidence remain unchanged. The six-study record contains324actual
controller fits/486,000primary updates;the trajectory bank is fixed.

The positive result is narrower than a deployable model:static image-region
features improve known-label cost routing,but do not establish a positive
completion bound,safe all-record support or a useful neural increment over
physical alternatives. All SOURCE inner-held data remain exposed development.
This is not an eight-page formatted CVPR submission,final anonymized release
or independent confirmation.
Private author administration is excluded from this anonymous package.

- [English manuscript](manuscript.md)
- [Six-study evidence,intervals and moment-error analysis](../feature_cost_evidence_20261029.md)
- [Static-scene results and limitations](../european_static_scene_cost_v1/readout.md)
- [Model card](../european_static_scene_cost_v1/model_card.md)
- [Next distinguishing experiment](../european_static_scene_cost_v1/next_action.md)
- [Chinese local/CREATE operating guide](../evidence_manuscript_v6/reproduction_zh.md)

## Reproduction

From the repository root,use the native arm64 environment. Existing completed
experiments are replayed,not overwritten:

```bash
.venv-pytorch/bin/python scripts/run_m3w_shared_reference_cost.py verify
.venv-pytorch/bin/python scripts/run_m3w_static_scene_cost.py verify
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_shared_reference_cost_v1 --verify
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_static_scene_cost_v1 --verify
.venv-pytorch/bin/python scripts/report_m3w_feature_cost_evidence.py --check
.venv-pytorch/bin/python scripts/build_m3w_evidence_manuscript_v7.py --check
```

The first two commands re-infer frozen policies using ignored licensed source
assets and private checkpoints. Bundle verification checks bytes only. The
feature export reuses utility intervals and recomputes descriptive moment-MSE
bootstrap arithmetic;the manuscript exporter is document verification,not new
training or new independent science. Public aggregates alone are not a complete
raw-data rebuild. CPU4/interop1/workers0,checkpoint/heartbeat/resume remain in
the registered runners. No new CREATE job was required for these local fits.

The displayed moment-MSE comparisons concern the mixed bank,where all seven
actions are admitted. Historical calibration arrays also contain seven-action
predictions for restricted banks;their excluded-action estimates are diagnostic,
not evidence of performance on actions those banks can deploy.

Raw pixel/frame detector tracks and static region proxies are not verified
metric geometry,effective seconds,human gold,true3D or foundation capability.
No deployment promotion,Stage5C execution orSMC. Submission readiness remains
false;formal submission still requires the user's final confirmation.
