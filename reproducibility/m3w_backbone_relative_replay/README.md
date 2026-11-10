# Replay the OpenDD relative-trajectory comparison

Extract the archive and run `python verify.py` with standard-library
Python. The verifier checks file digests, rebuilds the support counts
and paired location intervals from anonymous location-by-seed
aggregates, and verifies the inherited evidence archive. It does not
recompute model inference or read target trajectories. No raw data,
recording identifiers or weights are included.
