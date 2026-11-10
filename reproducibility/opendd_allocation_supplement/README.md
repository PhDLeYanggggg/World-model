# Forecast replacement: supplementary evidence

Run `python3 -I -B -S verify.py` after extracting this archive. No network,
model checkpoint, external library, or trajectory file is needed. The replay
first verifies the original supplement, then reconstructs the tables and
paired location intervals from anonymized OpenDD scalars. The original
reading guide and technical appendix are in `focused_evidence.zip`.

## OpenDD development: actual editing budgets

There are 202 recordings, seven locations and three training seeds. The
frozen online count head passes both references and the original risk checks
in 57 of 606 recording-seed views. With exactly its counts and no more than
its sum of per-edit peak shifts from constant velocity, hindsight still
finds 210 passing plans. Reordering the frozen head's own scores offline,
without labels, finds 60. The hindsight optimization is not a policy.

The additional experiment fits gain-only and gain-and-risk heads from
other locations. No configuration is chosen on the scored locations. The
gain-and-risk model with the penalty passes 60 views online using 1,292
edits, versus 57 using 780 for the frozen head. At matched realized counts
and movement caps, that model passes 42 versus 60 for the frozen scores.
This is evidence against deploying the new head. The same head without its
decision penalty passes 57 under the matched budget. A small number of
unresolved allocations retain the original online plan, with counts in
`allocation.json`. None of these results is external confirmation.

The scalar replay verifies aggregation, matched edit totals and interval
calculation. It does not recreate model fitting, per-view trajectory risk
certificates, or source labels. The 3,000 original resampling draws are
included so the replay does not depend on a particular random library.
