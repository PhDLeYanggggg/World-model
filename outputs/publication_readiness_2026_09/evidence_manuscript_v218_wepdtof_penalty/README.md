# Manuscript v218: distinguish the two pair-cost ablations

WEPDTOF's comparison removes the entire proximity-weighted relative
displacement penalty. Moving uses a squared relative score with a
cross-term-only control. Songdo uses separate learned donor heads.
v217 conflated these mechanisms. v218 corrects the methods, results
and supplement without
changing the WEPDTOF counts, gains, or missing-label envelopes.

The frozen-score replay explains the WEPDTOF tie. Thirty queries
have only one edit. In ten two-edit queries, the winner already
minimizes the penalty. The other 158 cannot switch at 0.10 because
their first possible crossover is at least 0.11013. No target
labels were opened, and this is not a score at a new weight.

The local matched files are under
`data/stage_cvpr2027_experiments/submission_admin/local_handoff_v218/`.
The title, abstract and disclosure are unchanged. The archive uses
the corrected appendix, replays the v217 numerical result without
its old wording, and includes the earlier v216 evidence. It contains
no raw trajectories or weights. Nothing was uploaded or submitted.

```sh
python scripts/build_m3w_wepdtof_penalty_candidate.py --compiler "$TECTONIC"
```

[Diagnostic](../wepdtof_penalty_breakpoint_v1/README.md) |
[Build receipt](receipt.json) | [Replay receipt](review_receipt.json).
