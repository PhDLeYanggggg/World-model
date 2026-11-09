# New paths at the same intervention counts

I tested whether the lower-coverage joint policy needed a better path
bank rather than another gain ranker. The expanded bank adds nine
past-only paths around OLS, varying position reanchoring and bounded
longitudinal or lateral velocity changes. Both banks keep the original
21,242 prescribed edits matched exactly by query, the same full-recording
2% risk check, and the same CV and OLS4 accuracy references.

| Outcome-aware search result | Original paths | Expanded paths |
| --- | ---: | ---: |
| Verified: original risk and positive conservative gain over both references | 10 | 10 |
| Nonpositive upper bound | 36 | 36 |
| No edits permitted | 21 | 21 |
| No selectable easy reference | 3 | 3 |
| Unresolved | 2 | 2 |

The two banks found positive witnesses for exactly the same views. New
paths account for 3,339 actions among the expanded witnesses, and the
returned maximin gain increased in nine of the ten common verified views.
The solver returned a weaker witness in the tenth view under its time
limit; the old solution is still in the expanded bank. This is a measure
of paths found, not a proof of equality or optimality. The paired
source-balanced difference in discovered passes is 0 pp on these data.

## What limits the gain?

For all 36 certified nonpositive views, even an OLS4-only objective with
the risk constraints removed remains nonpositive if each query keeps
its edit count. Allowing those same edits to move anywhere within the
recording makes the OLS4 upper bound positive in **8/36**, not a verified
gain. In **28/36**, even that more permissive bound remains nonpositive.
This separates the constraint on where edits happen from the total
number of edits and the quality of the available paths. I would test
more capable causal forecasts and query allocation before fitting
another ranker to this bank. The 2% deployment rule is unchanged.

## Scope and Reproduction

The solver saw outcomes; the candidate generators did not. The nine
development sources, 24 recordings, three seeds and 72 recording-seed
views were already exposed. The 3,000 bootstrap replicates resample
source groups, not overlapping windows. The task uses eight observed
positions and twelve annotation steps in local pixels. It does not
establish seconds, meters, physical safety or deployment readiness.

The formal search and capacity decomposition are `fresh_run`. Frozen
counts and role bindings are `cached_verified`. All positive witnesses
replayed under the original scorer. Runtime was 184 seconds locally with
about 5.2 GiB maximum RSS. The earlier run hit a presolve timeout bug and
was not pooled. No new neural model was trained, and neither DroneCrowd
calibration nor independent confirmation was run for this diagnostic.

[Protocol](protocol.md) | [Readout](readout.json) |
[Budget location](location_readout.json) |
[Working paper](../evidence_manuscript_v147/README.md)
