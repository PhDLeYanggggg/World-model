# Supplementary material

Start with `reading_guide.pdf` for the risk definition, external evaluation,
and matched forecast comparisons. `technical_appendix.pdf` contains the
extended methods and experiments.

Run `python -I -B -S verify.py` in this directory to check the bundle and
recompute the key counts and intervals. The standard library is sufficient.
`reviewer_evidence.zip` and `moving_peak_certificate.zip` also work
independently: extract either archive and run the same command there.

The statistical replay uses anonymous frozen summaries. The MovingDroneCrowd++
certificate independently solves the relaxed peak-budget allocation from grouped
errors. Neither task trains a model or reads raw labels. The OpenDD and
MovingDroneCrowd++ results are development evidence; the frozen WEPDTOF
comparison does not establish a deployable policy.
