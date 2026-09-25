# Source-Only Data Card

EuropeanSquares detector-derived trajectories use 8 observed and 12 predicted
annotation steps at raw stride12, in image pixels. Metric geometry, physical
timing and human-gold labels are not established. Missing future annotations
remain masked and are never an inference feature.

| Source roster | Localities | Rows |
|---|---|---:|
| A0 | 074, 082, 112, 126 | 116,823 |
| A1 | 007, 110, 119, 124 | 23,762 |
| A2 | 008, 020, 048, 067 | 178,384 |

The source store contains 318,969 rows across 12 localities. Producer A,
controller B and readout C use disjoint rosters in each fitted chain. Every
ordered A/B assignment is used, leaving the remaining four-locality roster
as C. Three seeds and two forecast pairs give 36 readout groups. These roles
overlap across settings; they are not 36 independent held-out datasets.

B-only labels determine sampling probabilities and regression loss. B-only
statistics determine feature and cost normalization. Candidate forecasts and
the existing utility heads are cached_verified against parent manifests.
Future endpoint, central velocity, test endpoint goals and future target
latents are not inference inputs. Registration binds source hashes and
upstream identity. Training/checkpoint replay verifies row and sampling
alignment; outcome-coordinate checks are separate from schema checks.

Source C is excluded from the present producer and controller chains but was
historically opened development data. Its bootstrap intervals quantify these
source comparisons, not a fresh confirmatory discovery. The six opened
selection localities (087, 092, 093, 103, 104, 125) are not evaluated this round.
The 12 reserved calibration and six confirmation localities remain closed.

The training-support audit counts dependent locality/role/seed views. The
evaluation averages seeds within locality, then resamples the four C
localities 3,000 times. Overlapping windows are not independent bootstrap
units. No raw trajectories, feature stores, histories, weights or third-party
images are included in Git. This is not SDD metric evaluation, seconds-level
forecasting, human gold, physical-safety certification, true3D or foundation
evidence. Stage5C and SMC remain off.
