# ACME CPU Detector Repair Registration

This is the CPU reference for a fixed-window comparison of the original
MobileNet-FPN detector against COCO-pretrained ResNet50-FPN v2.
The [registration](registration.json) binds the old camera schedules,
frame clocks, post-detector association and matching code, and weights.
Only the detector architecture and pretrained weights change in this
comparison; there is no fitting or threshold search.

The native arm64 CPU producer stopped after writing five CMU window
checkpoints. The first window took 103.391 seconds. No full CPU readout
was run. The five checkpoints are preserved in an ignored directory and
serve as a numerical overlap check for the separate
[CPU/MPS run](../acme_detector_hybrid_v1/README.md). They are not used to
fill missing windows in that run.

On a compatible local machine, with the registered videos and pinned
weights already available, resume the CPU producer with:

```sh
.venv-pytorch/bin/python scripts/run_m3w_acme_detector_repair.py produce --resume
```

Do not run the readout against an incomplete cache. This is an exposed
development diagnostic, not independent confirmation. No deployment,
Stage5C execution, or SMC.
