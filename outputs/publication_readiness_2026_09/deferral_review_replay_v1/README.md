# Reviewer Replay for the Timing Studies

The local ZIP contains six files, including alias-indexed aggregate
statistics and a standalone replay script. A fresh isolated Python
process recomputes recording-view totals and 32 paired confidence
intervals across four studies. The calculations average three seeds per
location, then perform 3,000 paired resamples of seven locations. Undefined
and negative results are retained.

```sh
.venv-pytorch/bin/python scripts/build_m3w_deferral_review.py
.venv-pytorch/bin/python scripts/build_m3w_deferral_review.py --check
```

Extract `data/stage_cvpr2027_experiments/deferral_review_replay_v1/review_bundle.zip`
locally and run `python replay.py` with the included pinned NumPy
dependency. No raw data, checkpoint, repository checkout or CREATE
account is needed to replay. The included scalars come from the frozen
results; the bootstrap is actually recomputed. The archive SHA-256 is
`46e55cb0f6e033867f4cd270e8dfc36f3045f7da10ed41e484ee01d01fcd14cc`.

This is not full experimental reproduction, independence verification
or certified anonymity. It does not qualify a policy for deployment
or unseal confirmation. [Machine receipt](verification.json).
