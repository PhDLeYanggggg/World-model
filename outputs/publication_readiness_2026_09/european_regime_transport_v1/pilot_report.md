# Native Reproduction and Crossed Pilot

Result source: fresh_run, native arm64 Torch CPU4/interop1/workers0.
PID68781 completed on2026-09-27 at03:07:17UTC.

The first original three-site cost-only head was retrained for2000 updates.
Every model parameter exactly matches the cached native control. Sampler
draws, fixed batch, preprocessing and non-easy loss scales also match.
This is a real-data training check, not import-only or NumPy fallback.

The first crossed two_cut3 head then completed200 updates, to be resumed
within its fixed2000-update budget. Its fixed diagnostic cost objective
changed from0.665513 to0.579733. Unknown-label draws:0. There are24901
parameters. No held outcome was used for training or this check.

Pure crossed-pilot fitting took0.202550 seconds. Linear extrapolation to
the registered1,728,000 updates is1750.03 seconds, excluding ancestry
verification, data loading, checkpoint handling and evaluation. It is an
estimate, not completed runtime. Free disk at pilot completion:13.739GiB.

Private pilot receipt:
`data/stage_cvpr2027_experiments/european_regime_transport_v1/pilot.json`.
Native bridge completion receipt SHA256:
`28f5bc87443f8e7f206b77a1985d67f625856739887cc3a7698c1ff1c911bf54`.
Registration commit71a591da and support commita729324d preceded the pilot.

No scientific result, independent evaluation or model promotion is claimed.
Obs8/pred12 native steps, detector pixels. Stage5C and SMC remain disabled.
