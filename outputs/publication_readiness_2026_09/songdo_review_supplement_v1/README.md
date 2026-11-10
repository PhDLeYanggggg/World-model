# Replaying the Songdo confirmation

This addendum accompanies the eight-page v170 candidate. I left the previous
review archive intact and included it in the updated bundle. The new material
comes from the frozen scoring receipts; no trajectories or future labels were
reopened for this build. The new computation is an offline replay, not a new
confirmation experiment.

The calibration used 40 flights in three linked source components. The
confirmation used 160 different flights in 14 components. The 480 views repeat
those flights across three seeds; the interval resamples the 14 components,
not the 480 views. The components are from one city and acquisition day.

Under the same two-edit budget, the joint minus independent conservative gain
was -0.000442 percentage points, with a 95% source bootstrap interval of
[-0.001499, +0.000237]. The additive comparison also spans zero. The frozen
joint rule released no edits. The independent rule released 15 views, of
which five met all original 2% checks and four also beat both references.
Fallback views do not count as passes. This result does not support deploying
the joint policy or claiming physical safety.

## Run the replay

Extract the locally built `review_supplement_v2.zip` and run `python3 -I -B -S
verify.py` from the extracted directory. It checks the hashes and runs both
the original evidence bundle and the anonymous Songdo replay. Songdo recomputes
thresholds, conservative contrasts, ADE/FDE summaries, bootstrap intervals
and risk counts from scalar receipts. It does not rerun inference, reconstruct
tracks or establish whether the original labels were correct.

To rebuild the archive in the project with its private scoring receipts and
the verified v1 bundle present, run `python3
scripts/build_m3w_songdo_review_supplement.py` in an environment containing the
project scoring dependencies plus ReportLab and pypdf. The builder checks all
confirmation score hashes and the pre-outcome threshold freeze before writing
the archive, the one-page PDF and [the receipt](receipt.json). The archive and
PDF remain local; neither was uploaded to OpenReview.
