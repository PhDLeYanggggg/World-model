# External forecast replacement at matched budgets

This comparison puts three frozen evaluations side by side. It does
not pool cohorts, change the original 2% checks, or rejoin future
positions. Each cohort uses eight observed and twelve forecast steps.
WEPDTOF and Songdo match the original edit count and peak departure;
Athens originally matches count. The sensitivity holds the same
recipients and directions and matches departure from CV step by step.

| Cohort | Source units used | Original envelope (pp) | Equal steps envelope (pp) |
| --- | ---: | ---: | ---: |
| WEPDTOF | 10 of 13 sources | [-0.04501, +0.01760] | [-0.04181, +0.01829] |
| Songdo | 14 of 14 source components | [-0.000442, -0.000102] | [-0.000423, -0.000096] |
| Athens | 4 of 4 acquisition dates | [-0.04427, +0.05304] | [-0.03823, +0.04511] |

These are joint-minus-independent gains in points of observed CV
error; the envelopes vary the missing positions within the scored
sample. They are not confidence intervals. In Songdo the sample
envelope favors independent selection. The paired source interval
for its equal-step lower endpoint is [-0.001317, +0.000122] pp,
so this is not a population-level joint disadvantage.

The additional calculation pairs the **observed** gain before and
after step matching within each source unit. It uses 3,000 draws,
with a common bootstrap procedure but separate resampling by cohort.

| Cohort | Change in observed gain (pp) | 95% paired interval (pp) |
| --- | ---: | ---: |
| WEPDTOF | +0.001951 | [-0.001834, +0.005692] |
| Songdo | +0.000013 | [-0.000108, +0.000170] |
| Athens | -0.000944 | [-0.001524, -0.000365] |

Matching steps reduces the small observed Athens advantage, but
the full-grid contrast still includes zero. Its resampling is
descriptive of four dates in one city. Seeds and overlapping windows
are not additional independent source units.

| Cohort | Fixed joint proposals passing risk and both utility checks | Actual joint release |
| --- | ---: | --- |
| WEPDTOF | 0/45 | Zero edits |
| Songdo | 53/480 | Zero flight-seed views |
| Athens | 1/18 | Not evaluated |

The original independent Songdo policy released 15 flight-seed
views; four passed risk and both utility checks. Proposal checks are
not releases, and zero edits do not establish a risk pass.

## Manuscript takeaway

> Matching departure at each prediction step does not establish
> an external advantage for joint over independent replacement.
> The fixed proposal comparison and the calibrated release test
> must be reported separately: some Songdo proposals meet the
> outcome checks, but the frozen joint rule does not select any
> for release.

The one-page anonymous PDF is under
`data/stage_cvpr2027_experiments/submission_admin/external_budget_comparison_v1/`.
It is a separate insert. The author's Word and existing v218
handoff are not modified.

~~~sh
python scripts/audit_m3w_external_budget_comparison.py
python scripts/build_m3w_external_comparison_page.py
~~~

[Source-level replay and hashes](readout.json) |
[PDF receipt](render_receipt.json).
