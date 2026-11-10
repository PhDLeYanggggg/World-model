# Forecast replacement: evidence and replay

Start with `reading_guide.pdf`. It explains the displacement bound, the
choice of recipients and the distinction between matched proposals and
actual release. `technical_appendix.pdf` and `songdo_confirmation.pdf`
contain the detailed analyses.

Extract this archive and run `python3 -I -B -S verify.py` in its root.
The program uses only the Python standard library. It verifies each
member and reproduces the Moving budget ceilings, the OpenDD counts,
the WEPDTOF completion bounds and the Songdo thresholds, contrasts and
release counts. No network, tracking data or model weights are required.

This is a replay of frozen aggregate evidence, not a reproduction of
dataset conversion, training or trajectory inference. A nonnegative
hindsight ceiling is not a deployable forecast. Zero edits do not pass
the original recording-level risk checks.
