# Replay the OpenDD fixed-spend uniform comparison

Run `python verify.py` after extracting this archive. Only the Python
standard library is required. The verifier checks the archive hashes,
recomputes the expected pass total and location-bootstrap intervals
from aggregate location-by-seed data, and replays the inherited
fixed-spend and external results. It does not rerun forecast inference
or the uniform sampler. No targets, recording IDs, or model weights
are included. The reader PDF is unchanged from the prior archive.
