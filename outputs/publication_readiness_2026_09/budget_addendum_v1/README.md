# Matched-budget diagnostics in the supplement

I added two English pages to a separate local copy of the supplement.
The first places exact-budget hindsight capacity next to learned performance.
The second explains why moving the same edit slots can remove the reference
needed to estimate risk, and why protecting that reference is not enough
to improve prediction. The diagnostic is development evidence from OpenDD,
not independent confirmation or a new result from a trained policy.

The two pages use the [exact-slot bound](../opendd_exact_slot_headroom_v2/README.md),
[expanded menu](../opendd_self_menu_v1/README.md),
[refitted costs](../opendd_expanded_costs_v1/README.md),
[paired-window budget study](../opendd_paired_window_budget_v1/README.md), and
[easy-reference guard](../opendd_easy_slot_support_v1/README.md).
The builder checks the six bound evidence files before using their numbers.

| Local artifact | SHA-256 |
| --- | --- |
| `supplement_with_budget_addendum.pdf` | `24feba604f08cd4925cf9828f01035b4f69e9520168af7c4e27f23971ae5a5ea` |
| `matched_budget_evidence_replay.zip` | `5c71826115f7a71c5fc19c4e72208b1a1e97efa1dae6e6863f9f48039dbe8ef7` |

The source guide edition and working manuscript are unchanged. The first
52 pages of the 54-page copy are pixel-identical at 72 DPI. The 275 existing
named destinations and nine guide links keep their targets. I inspected pages
53 and 54 as images. They add no external links or author identifiers.
The PDF and archive both rebuilt byte for byte. An extracted copy ran the
table and bootstrap replay outside the repository with the standard library.
The replay starts with anonymous source-level summary values; it does not
retrain models or replay upstream predictions and labels.

```sh
# Use an environment containing numpy, pypdf and reportlab.
base=data/stage_cvpr2027_experiments/submission_admin
python scripts/build_m3w_budget_addendum.py \
  "$base/supplement_with_budget_addendum.pdf" \
  "$base/matched_budget_evidence_replay.zip"
# In the extracted replay directory:
python -I -B -S verify.py
```

The builder refuses to overwrite either output; use new paths to reproduce
an existing copy. No PDF, archive, raw data, weights or author records are
in Git. Nothing was uploaded. The [CVPR 2027 dates](https://cvpr.thecvf.com/Conferences/2027/Dates)
list Nov. 16 AoE for the paper and Nov. 23 AoE for supplementary material.

The remaining scientific gap is a supported, independent comparison of the
joint and additive policies. The existing WEPDTOF frozen result is not
positive and the DroneCrowd confirmation partition remains unopened.
Packaging these diagnostics does not resolve that gap.
