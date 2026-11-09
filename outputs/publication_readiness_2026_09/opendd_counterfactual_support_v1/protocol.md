# Protecting selected easy reference under query replacement

Use the v152 frozen outer roster and the same analytic-plus-spline
candidate bank. Every arm preserves each original query's count and
the multiset of peak displacements, including zero-edit recordings.
At most one query per recording may be replaced, at the first current
query with an admissible candidate. All other edits remain original.
Do not look at future queries, even to find a better replacement.

For each alternative query plan, predict its conservative paired
utility against the unchanged anchor and a binary loss of easy
reference. Both models have the same features, draw stream, loss and
architecture. In the local model, the binary label is loss of every
original easy reference inside the query. In the whole-record model,
the label is loss of all selected easy reference in the recording
under this single replacement. The latter accounts for unchanged
anchors before and after the query only in the training label. No
outcomes or future queries appear in features. Donor slots and
forecasts exclude donor and outer locations as in v152.

Both gates use predicted lower gain above 0.005 in history-motion
units and estimated reference-loss probability at most 0.20, fixed
before scoring, not independently calibrated. A gain-only arm uses
the whole-record model but omits the loss gate. Also compare it on
exactly the recordings and query times where the whole-record model
acts, keeping the same number of query replacements. This matched
attribution is not a standalone online policy. Rejecting a replacement
keeps all original analytic edits; it does not imply zero risk.

Primary: whole versus local loss training, actual reference losses,
original risk/dual passes and direct paired conservative gain. Compare
also with past prefix and the same-query gain control. Report all views
and three seeds with 3,000 paired location draws. After the freeze,
diagnose best single-query hindsight opportunity in this same bank.
Original 2% rule unchanged; no independent confirmation access.
