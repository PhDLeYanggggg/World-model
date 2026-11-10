# Three Songdo case studies

`failure_cases.pdf` illustrates recipient choice, path choice with fixed
recipients, and an observed gain whose sign is unresolved when labels are
missing. The figure uses the previously scored cohort. It does not change
the primary result or the 2% release rule. Shading is a missing-label bound,
not a confidence interval. `protocol.md` gives the case-selection rule.

Run `python verify.py` to check the figure hash, edit membership, support
masks, cumulative summaries and inherited statistical replay. This checks
the anonymized arithmetic; it does not reconstruct trajectories, retrain
models, or repeat case selection from raw labels. No flight identity or
absolute coordinates are included.
