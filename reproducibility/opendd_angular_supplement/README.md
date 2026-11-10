# Direction choices at fixed displacement

Extract the archive and run `python3 -I -B -S verify.py`. This requires
no network, model weights, trajectories, or third-party libraries.
It retains the preceding evidence and independently recomputes
counts and location intervals from anonymized grouped scalars.
It does not reproduce training or create new external evidence.

All new panels are OpenDD development results at 606 recording-seed
views, 780 edited rows, and the original 2% risk rule.

- Direction diversity expands the exact dual-pass frontier from 181
  to 216. It does not establish a policy.
- The learned direction on the original recipients reaches 77 dual
  passes versus 60, but does not establish conservative gain.
- Thirteen angles, including the original, have a hindsight
  utility of +0.05573 percentage points.
- The zero-threshold factorized angle choice rotates 710 rows.
  Cross-fitted donor calibration cuts this to 198; requiring
  nonnegative utility at each donor yields no rotations.
- At those same 198 rows, a donor-constant angle outperforms the
  learned angle in direct conservative utility. Only 44 views
  allow a different placement at identical counts. Exact random
  placement with the constant angle recovers almost all utility.

The grouped comparisons use 3,000 paired location resamples after
averaging seeds. No submitted or deployed model is implied.
