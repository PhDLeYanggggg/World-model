# Matched normalization repair

The first run observed losses above 100,000 where OpenDD training
examples had near-stationary prefixes and nonzero future displacement.
Preserve the original checkpoints and scores. Repeat the same three
arms and seeds from scratch with a single change: for the loss, lower-bound
each history RMS scale by the 10th percentile of positive RMS scales in
the unperturbed donor histories. The feature calculation, draw order,
augmentation, weights, model size, optimizer and update count are unchanged.

This is a local training repair, not a different scoring metric. Use the
same 8-to-12 camera pilot, CV/OLS4 comparisons and original 2% checks.
All results are exposed-development results. Select no new deployment or
threshold, and do not open non-pilot external outcomes.
