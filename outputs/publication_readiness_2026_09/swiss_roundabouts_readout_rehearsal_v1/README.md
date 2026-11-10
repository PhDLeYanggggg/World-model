# Swiss scoring: keep two-actor windows in the risk denominator

I exercised all 54 frozen file-seed packets from the
[Swiss pilot](../swiss_roundabouts_source_intake_v1/README.md) with
fabricated future positions and masks. The 258 unmatched views are
exactly the three-seed repeats of 86 two-actor windows. The inherited
candidate menu needs at least three actors; I did not relax it.
The other 7,431 views keep two distinct edits at the same peak radius.

The implementation treats the two estimands separately. The same-budget
comparison uses the common matched queries only, and returns undefined
if none exist. The full-roster path retains CV in unmatched queries,
keeps all actors for the original 2% recording rule, and does not treat
an empty comparison or a fallback as certification. With two actors
the window is still represented, even though it cannot enter the
matched-proposal subset.

A separate stream-level target join checks original actor indices,
clock-segment boundaries and missing labels. Its tests use artificial
CSV rows. No command in this rehearsal opens Swiss future coordinates,
and there is no registered Swiss evaluation. The [receipt](receipt.json)
binds all frozen packets and records the exact support counts, not
metrics on artificial outcomes.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/rehearse_m3w_swiss_roundabout_readout.py
```
