# Forecast replacement: evidence map

I reorganized the argument around three questions rather than the order in
which the experiments were run. This is a separate five-page local build.
The title, abstract, disclosure and 17 reference entries carry over from
v171. The two main vector figures are unchanged. Older drafts and the
review archive remain intact.

| Question | Evidence | What it does not show |
| --- | --- | --- |
| Can the displacement budget support an improvement? | [Moving completion bound](../moving_completion_envelope_v1/README.md): 67/72 excluded under the conservative score, 36/72 on observed steps, 9/72 even with favorable missing positions. | A nonnegative ceiling does not identify a valid, past-only replacement. |
| Can a policy find the useful replacements? | OpenDD development: 330/606 label-aware feasible plans under the strict menu and chronological credit; 57/606 passes for the past-only count controller. The two retained figures separate search capacity from realized screening. | Neither a hindsight solution nor an extra seed is an independent test. |
| Does the release rule transfer? | [WEPDTOF](../wepdtof_missing_label_interval_v1/README.md): 0/45 dual passes. [Songdo](../songdo_calibrated_20261110/README.md): 0/480 joint releases; 5/15 independent releases pass risk. | Zero release is not a risk pass. DroneCrowd confirmation is still sealed. |

The Moving primary comparison matches edit count, while the Songdo comparison
also matches peak displacement. Both keep the original CV-relative 2% checks.
No new experiment or model selection is implied by the rewrite. The abstract
is preserved as requested; before final author approval, its 67/72 sentence
should explicitly say that it refers to the conservative missing-label score.

## Local rebuild

```sh
python scripts/build_m3w_focused_candidate.py --compiler /path/to/tectonic
```

The command uses the local `focused_body.tex`, the hash-verified v171 archive,
the original review style, and the tracked aggregate receipts. It checks the
page count, citations, disclosure, selected numeric markers and PDF anonymity.
See the [build receipt](receipt.json). The private draft PDF, source and archive
are not in Git. The [v3 anonymous review archive](../focused_review_supplement_v1/README.md)
pairs with this draft and replays all three Moving bounds and the WEPDTOF
completion interval as well as the earlier evidence.
