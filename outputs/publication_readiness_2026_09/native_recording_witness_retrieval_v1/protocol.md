# Retrieval From Positive Recording-Level Witnesses

Hold the full query, forecast bank, original intervention cap, eight
recipient slots, decoder, cost fit and gate fixed from the gain-control
experiment. Change the second half of the recipient shortlist: rank it
by a ridge model trained to predict inclusion in a verified recording
witness that satisfies the original CV risk rule and achieves positive
conservative utility versus OLS4. No threshold selection.

One teacher is solved separately per exposed recording and forecaster
seed. For this teacher only, allow switches on labeled rows and reduce
each row to its minimum observed-error non-CV path. Optimize query-weighted
conservative OLS4 gain subject to the original query counts as upper
bounds and the recording risk/support constraints. Charge the unchanged
CV-vs-OLS4 disagreement on wholly missing rows. Give positive training
labels only to members of checked positive witnesses. An unresolved
solve may contribute a checked positive incumbent, not an optimality
claim; if none is checked it contributes no positive label. This is
a restricted witness producer, not the all-recipient capacity upper bound.

For each held-source retriever, the teacher labels from that source
are excluded from the sufficient statistics used for its fit. The
recipient ranks use past features only and are frozen before scoring.
Use the existing gain-control decisions and capacity checkpoints with
hash validation. Report the fixed-size shortlist's positive OLS4
capacity, exact gains and losses in feasibility, and 3,000 paired
source bootstrap draws (forecaster seeds averaged). Compare actual
original-gate outputs against CV, OLS4 and the unshortlisted control.
Use 34 exposed Moving A recordings, 14 sources and three forecaster
seeds, eight observed and twelve predicted annotation steps in pixels.
No external role access, new deployment, Stage5C or SMC.
