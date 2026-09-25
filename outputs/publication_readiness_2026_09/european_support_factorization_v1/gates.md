# Evidence Gates

Engineering completion is separate from method efficacy; no aggregate pass fraction is used as a model-success score.

| Gate | Status | Evidence |
|---|---|---|
| Registered fixed comparison | pass | Registration `8f2aca43` pushed before decisions; no threshold or producer refit |
| Decision-before-readout barrier | pass | Batch 14:54:57 UTC, fitting 14:59:24 UTC, first new readout phase 15:02:32 UTC on 2026-09-25 |
| Source/producer exclusion | pass, cached_verified boundary retained | Prior identities and training-only boxes hash-bound; no held labels used for decisions |
| Complete matrix | pass | 936 views, including 288 replay anchors and 648 new factor/control views |
| Independent arithmetic and replays | pass | 12,528 metric checks; 936 scalar decisions; 1,728 old metric matches |
| Removal attribution | pass | 4,608 exact decomposition equalities; unknown labels not zeroed |
| Scoped tests | pass | 326 tests in 53 files; full legacy suite not_run |
| Support-factor all-ADE lift vs stop | fail | No factor guard has a positive interval; most point estimates are negative |
| Support-factor hard-ADE lift vs stop | fail | No positive factor-guard hard interval |
| Equal-count ordering advantage | not established | Mixed risk/random comparisons; flexible-query and degenerate cases retained |
| Positive-easy empirical limit | pass on observed support | Worst degradation 0.43686%, below 2%; not a physical-safety guarantee |
| Zero-CV stopping repair | retained, limited support | Four unique rows, one locality, two labels each, no endpoint |
| Producer transport explanation | not_run | Producer models/training size were not changed |
| New independent confirmation | not_run | Reserved roles remain closed; opened sources cannot regain independence by renaming |
| New deployment / submission-ready method | false | Support filtering not promoted; evidence and matched public-method gaps remain |
| Stage5C execution | false | Prohibited |
| SMC | false | Prohibited |

Pixel raw-frame obs8/pred12 only; no metric, seconds, human-gold, true-3D or foundation claim.
