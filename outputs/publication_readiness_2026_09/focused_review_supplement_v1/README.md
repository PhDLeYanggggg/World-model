# Forecast replacement: offline review evidence

This archive pairs with the five-page local draft. I kept the existing
technical appendix and Songdo sheet and added a one-page guide that follows
the main argument. The older archives remain intact.

| Question | Verified replay |
| --- | --- |
| Can the budget support a gain? | Moving: 67/72 conservative exclusions, 36/72 on observed steps and 9/72 even under favorable missing labels, with pooled displacement. |
| Can a rule find those gains from the past? | OpenDD: 330/606 verified hindsight plans and 57/606 passes for the past-only controller under the corresponding restrictions. |
| Does the release rule transfer? | WEPDTOF: 0/45 dual passes and a joint-minus-independent completion interval of [-0.04501, +0.01760] pp. Songdo: joint 0/480 releases; independent releases 15, of which five pass risk and four pass both risk and utility. |

## Run the replay

Extract the local `review_supplement_v3.zip` and run
`python3 -I -B -S verify.py` from the extracted directory. It needs no network,
dataset, model weights or nonstandard Python package. The verifier checks the
archive hashes and rebuilds the counts and intervals from anonymous scalars.
The extra Moving data contains aggregated marginal-gain segments, not agent
paths, timestamps, source names or labels. Its view order and budget are bound
to the existing certificate. The WEPDTOF upper endpoint comes from the original
paired scores, not newly filled-in labels. Undefined sources are not counted
as zero gain.

The archive verifies the results of frozen analyses, not model training or
the validity of the original annotations. The Moving build re-aggregates
previously exposed development evidence locally. It does not reopen external
confirmation labels or change the original 2% rule. This is not a new policy
or a positive result for joint deployment.

## Rebuild locally

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python scripts/build_m3w_focused_review_supplement.py
```

Use the arm64 project environment with the scoring dependencies, ReportLab
and pypdf. The command needs the existing local development files, v172 draft
and v2 archive. It checks the inherited hashes, runs isolated replays and
writes a deterministic archive. The PDF and ZIP are local only; their hashes
and aggregate verification are in the [receipt](receipt.json). Nothing was
uploaded to OpenReview.
