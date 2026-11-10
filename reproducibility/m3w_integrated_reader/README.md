# Supplementary material and aggregate replay

Open `supplement.pdf` for the full reader appendix. It includes the
original development and external evidence, the fitting and scoring
recipe, the fixed-spend and equal-training controls, the relative
trajectory results, and the equal-coverage masking figure.

Run `python verify.py` from the extracted directory to check the
archive digests and replay the table values using standard-library
Python. This verifies published aggregates, not inference or training.
The archive is self-contained and contains no raw trajectories,
private recording identifiers, or trained weights.
