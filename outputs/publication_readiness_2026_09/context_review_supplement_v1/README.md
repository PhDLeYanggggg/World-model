# Anonymous supplement v8

The v179 main manuscript is unchanged. The new supplement retains the
v7 evidence and adds the OpenDD context ablation, including the exact
random control on the same 101 exchange views. It recomputes the counts
and 3,000-draw intervals from anonymous location and seed summaries.
It does not retrain the heads or contain their weights or raw targets.

The local archive is
`data/stage_cvpr2027_experiments/submission_admin/review_supplement_v8.zip`.
Extract into a clean directory and run `python3 -I -B -S verify.py` there.
The [receipt](receipt.json) binds it to the unchanged manuscript and both
readouts. Two builds were byte-identical and the isolated replay matched
the source summaries. Nothing was uploaded.
