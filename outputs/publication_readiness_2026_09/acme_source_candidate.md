# ACME: Source Candidate, Not Admitted Data

Checked 2026-09-16. `fresh_run` public-source review; acquisition, conversion,
training and evaluation are `not_run`.

The [author project](https://raoshashank.github.io/acme-socnav-dataset/) describes
BEV recordings at seven collection sites, human-verified tracks, homography-based
calibration and ETH/UCY-compatible 2.5 Hz exports. It also describes five-frame
smoothing and interpolation/removal of noise jumps. The inspected page did not
expose a dataset download/terms link; its website license is not dataset permission.

The [July 2026 preprint](https://arxiv.org/abs/2607.21964v1) reports 43.5 hours of
overhead tracking, states submission to IJRR, and is not an acceptance notice.
Its onboard duration differs from the current project page; release identity
needs checking rather than combining versions.

Section 11 of the [preprint full text](https://arxiv.org/html/2607.21964v1)
describes private reviewer access and a planned Hugging Face release under CC BY,
with processing tools planned under Apache 2.0. These are release intentions,
not a verified current downloadable artifact or license manifest.

Research judgment: prioritize checking annotation access and recording/site
identities before acquiring large videos. This could strengthen independent
cross-site and visual-context experiments. Verify whether smoothing uses future
positions before admitting any observation fields, and separate all site roles
before model scoring. Author-reported calibration does not verify our local files.
Do not treat institutions, cameras or overlapping clips as independent samples
without a provenance audit. Current M3W metric, seconds and deployment claims
remain unchanged.
