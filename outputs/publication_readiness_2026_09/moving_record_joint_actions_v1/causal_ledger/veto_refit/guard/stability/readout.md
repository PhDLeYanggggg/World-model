# Source-Stable Veto Costs Reduce Some Harm But Lose Utility

Fresh risk-head refits on exposed Moving A, not independent confirmation.
The task remains 8 observed and 12 future annotation steps in image
coordinates. Across 42 source-seed cells, seven sources fit each model;
the other seven, including the held source, do not. Three existing frozen
forecasters supply the candidate bank. No new forecaster is trained.

I fit a pooled veto cost head and seven heads that each omit one fitting
source. A proposed cost reduction shrinks toward zero by one jackknife
standard error. This measures sensitivity of the residual fit conditional
on the same donor proposals, not a full-pipeline source deletion. Upward
revisions are unchanged. The rule is fixed
before scoring and is not a risk guarantee. Both frozen controls reproduce
exactly, and all policies switch exactly 6,952 times, matched query by query.

| Policy | Full original passes / 102 | Undefined easy risk | Easy positive-harm upper >2% | Signed easy net >2% | Selected reopened actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| No reopening | 2 | 35 | 65 | 8 | 0 |
| Pooled cost revision | 1 | 39 | 61 | 15 | 2,531 |
| Source-stable cost revision | 4 | 37 | 59 | 9 | 822 |

The stable head changes 5,245 row-level action assignments relative to
the pooled head, counting both changes of candidate and reassignment
of switches to different rows. It reduces selected reopened actions
without reducing the overall switch count.

| Paired comparison | Completion lower difference (pp) | 95% source bootstrap interval |
| --- | ---: | ---: |
| Stable vs pooled (primary) | -0.18428 | [-0.34810, -0.04103] |
| Stable vs no reopening | -0.07070 | [-0.27405, +0.04120] |
| Pooled vs no reopening | -0.00829 | [-0.38190, +0.27738] |

The 3,000 draws resample whole sources after averaging the three
forecaster seeds. The primary interval concerns completion-lower utility
under incomplete outcomes, not the unknown complete-data point effect.
Deleting a source reveals cost sensitivity and makes some choices less
harmful, but this penalty loses useful interventions as well. Four full
passes and 37 undefined easy views are not deployment evidence.

The original 2% limits and support checks still apply. The study adds
no identity-fallback certificate. The scores replay byte-for-byte. No
independent outcome access, deployment change, Stage5C execution or SMC.
