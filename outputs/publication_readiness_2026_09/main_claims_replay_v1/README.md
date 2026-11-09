# Offline replay of the headline counts

I rebuilt the headline counts from the frozen study records and
checked them against the numbers generated into the working paper.
The archive contains only a small verifier, the result fields
needed for it, a manifest, and instructions.

| Evidence | Replayed result | Unit |
| --- | --- | --- |
| OpenDD hindsight capacity | 330 / 606, one unresolved | 202 recordings across three seeds; seven development locations |
| OpenDD past-only controller | 57 / 606 | Same menu and budget, not necessarily the same realized edits |
| MovingDroneCrowd++ relaxed OLS4 bound | 67 / 72 nonpositive | 24 development recordings across three seeds |
| WEPDTOF fixed-count joint proposal | 0 / 45 dual passes | 15 recordings from 13 frozen sources, across three seeds |

The OpenDD replay sums location-level outcomes, preserving the unresolved
case. The MovingDroneCrowd++ replay counts nonpositive upper bounds.
For WEPDTOF, it checks both positive gains and the saved risk decision.
The replay does not refit cutoffs or recompute the risk bound from labels.

```sh
.venv-pytorch/bin/python scripts/package_m3w_main_claims.py
```

The exporter checks four sources against the draft's build manifest,
replays the aggregate calculation in isolation, and checks eight paper
macros. It writes the local `main_claims_replay.zip` under
`data/stage_cvpr2027_experiments/submission_admin/`. Extract it
and run `python verify.py`; this command needs only Python's standard
library. Two exports had identical bytes, and the extracted archive
replayed without the repository. The SHA-256 of this revision is
`6d2f280fac09591ba7b0729755ded98967901db885f1e0d3d37464c24a12a8a6`.

This verifies reported arithmetic, not the training pipeline or
independence of the evidence. The paper, policies and experiment
results are unchanged. No raw outcomes, private checkpoints or
reserved confirmation data were added to the archive.
