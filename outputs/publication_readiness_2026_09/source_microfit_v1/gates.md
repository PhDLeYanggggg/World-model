# Microfit Evidence Boundaries

| Requirement | Status | Scope |
| --- | --- | --- |
| Registered controlled experiment | Pass | Two decoder laws, two selected training cohorts, three seeds; no score-selected replacement |
| Complete numerical run | Pass |12fits,24000updates; real arm64Torch |
| Can original head fit nonzero targets? | Yes on selected training rows | Mean training reduction98.18%; not held-source prediction |
| Can original head fit the selected mixed cohort? | Yes |96.60% mean training reduction; easy absolute harm remains positive |
| Is clipping alone a universal fitting obstruction? | Refuted for this microfit | Original clips every update and still fits; full-corpus cause not identified |
| Does rescaling repair the full benchmark? | Not run / unproved | No new held-source or full-corpus score; not every seed benefits |
| Generalization or deployment improvement | Not established | Training rows only; inherited60-model negative result unchanged |
| Exact replay and resume | Pass |12exact arrays, six matched pairs,37unchanged artifacts,zero newupdates on completed resume |
| Main roles/metric unchanged | Pass | No main fitting or sealed evaluation |
| Long-term research objective | Not complete | Useful predictor and independent joint-risk evidence still missing |
| Stage5C execution / SMC | False / false | Neither enabled |

Verdict: `training_fitability_verified_full_corpus_failure_unresolved`.
