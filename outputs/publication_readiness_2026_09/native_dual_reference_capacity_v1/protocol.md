# Can The Fixed Bank Beat Both Motion References Under The Original Rule?

This is a label-aware capacity experiment on the exposed Moving A development
roster, not a model or a test-set threshold selection. Keep the 34 recordings,
14 source groups, three frozen forecaster seeds, 8/12 native annotation
steps, the original 2% CV risk and reference-support rules, and the original
genuine replacement count in every query. Compare one frozen nomination per
person with the full forecast bank. Keep CV as fallback and risk denominator;
OLS4 is the second accuracy reference, not a new floor.

For each recording-seed view, maximize the smaller of the conservative
utility masses against CV and OLS4, with both computed on the full
past-eligible roster. Gains share the observed CV denominator, so the
smaller-mass sign is also the smaller normalized gain sign. On partly
labeled rows, retain the original available-step ADE. On wholly unlabeled
rows, subtract the maximum distance to the specific reference path. In
particular, an unselected row still incurs its CV-vs-OLS4 unknown debit in
the OLS4 contrast. The score is additive with one choice per person.
On observed rows the lowest-error legal replacement dominates for
both references and all risk constraints. Do not prune wholly unknown
actions by CV disagreement: OLS4 disagreement need not order them the
same way.

Use 3 seconds per solve, then up to 12 seconds when neither a verified
strictly positive witness nor a negative upper bound or infeasibility
settles the question. Re-score each candidate witness with the original
scorer and independently check genuine switch counts. Small signed
margins around solver tolerance remain unresolved rather than being
called positive or impossible. Separately report verified witnesses,
infeasibility of the original risk/count constraints, a negative
dual-reference objective bound, and unresolved views. A full-bank
witness from the fixed-nomination subset is also valid by construction.

The prior CV-optimized positive-OLS4 witness counts are lower bounds
on existence, not an OLS4-optimal ceiling. This experiment can distinguish
an avoidable choice of objective from a bottleneck in the forecast bank,
at the registered budget. Per-recording hindsight existence is not the
external bootstrap or deployment gate. No action labels, raw predictions
or protected data are exported. No independent outcomes, threshold changes,
Stage5C execution, SMC, or metric/seconds claims.
