# Hurdle-Risk Gates

25 September 2026. Gates below describe this source-development experiment;
passing engineering checks does not establish independent safety or readiness.

| Gate | Status | Evidence / limit |
|---|---|---|
| Registered controlled comparison | Pass | `7a1cd0d9` precedes new fitting; architecture, draws, budgets and risk rule fixed |
| Causal inputs and producer exclusion | Pass within audited scope | Prior identity chain verified; no future mask in rollout bound; labels only in loss/accounting |
| Real Torch training | Pass | 72 heads, 144,000 updates, 3 seeds, native arm64; not a NumPy fallback |
| Checkpoint and sampler replay | Pass within stated scope | 72 heads, first 4,096 held-index rows/head, 72 exact sampler matches |
| Full metric/control reproduction | Pass | 144 views and 72 prior control views match |
| Scoped software regression | Pass | 218 tests / 34 files; full legacy suite not run |
| Positive-easy preservation | Pass observed | All 18 neural hurdle views <=2%; maximum 0.6746% |
| Zero-CV preservation | Fail | All 12 views with zero-reference readout rows harm some; other six have no such rows |
| Stable neural all-ADE advantage over matched damping | Fail | 0/18 positive conditional intervals, 15/18 negative |
| Stable hard-subset advantage | Fail | 2/18 positive intervals in one fold, 15/18 negative; no zero-reference support in the positive fold |
| Occurrence signal beyond fitting prior | Partial diagnostic | Neural Brier gains in 9/9 easy and 6/9 all views; not intervention or dynamics proof |
| Independent calibration / confirmation | Not run | Reserved roles remain closed; current localities are all development-exposed |
| New deployment / submission readiness | False | Joint evidence requirements remain unmet |
| Stage5C execution | False | Prohibited and not run |
| SMC enabled | False | Prohibited and not run |

Observed safety combines positive-easy preservation and zero-reference harm.
It is not a physical-safety certificate, a conformal guarantee or a test of
all possible rare strata. Eight excluded localities per fit share a twelve-
locality development pool. Bootstrap intervals use 3,000 locality resamples;
they are conditional and not multiplicity adjusted. No risk tolerance changed.

Image-pixel released detector tracks, 8 observed / 12 predicted raw-stride-12
steps only. No metric, seconds, human-gold, true-3D or foundation claim.
