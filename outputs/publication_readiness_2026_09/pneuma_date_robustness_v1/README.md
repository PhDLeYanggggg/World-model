# Athens results by acquisition date

On the frozen Athens roster, both joint and same-family replacement
improve on constant velocity at the lower missing-label bound on
every observed date. Neither the same comparison with independent
selection nor matching movement at every forecast step establishes
a benefit attributable to joint selection.

| Contrast | Equal-date missing-label envelope (pp) | Dates definitely positive / negative / unresolved |
| --- | ---: | ---: |
| Joint vs. CV | [+0.02383, +0.08858] | 4 / 0 / 0 |
| Same family vs. CV | [+0.02411, +0.08999] | 4 / 0 / 0 |
| Independent vs. CV | [+0.00387, +0.09977] | 2 / 0 / 2 |
| Additive vs. CV | [+0.00197, +0.10230] | 2 / 0 / 2 |
| Joint vs. independent | [-0.04427, +0.05304] | 0 / 0 / 4 |
| Joint vs. additive | [-0.04448, +0.05263] | 0 / 0 / 4 |
| Joint vs. same family | [-0.00156, -0.00012] | 1 / 3 / 0 |
| Joint vs. independent, equal movement at each step | [-0.03823, +0.04511] | 0 / 0 / 4 |

All leave-one-date-out summaries retain the positive lower bound
for joint and same family against CV. In fact, the lower bound
cannot fall below +0.00432 and +0.00434 pp, respectively, under
any nonnegative reweighting of these four date summaries. The
joint/independent envelope crosses zero in each date and each
leave-one-date-out summary, including the step-matched comparison.
The small equal-date disadvantage against the same-family control
does not extend to every date.

This is a secondary diagnostic on the observed sample, not a new
test cohort or a population interval. Reweighting dates cannot
simulate a new acquisition setting. The original bootstrap and
the unchanged 2% rule remain in the
[registered readout](../pneuma_confirmation_v1/README.md). Fixed
joint proposals passed the original risk-and-dual-utility check
in 1/18 member-seed views. No Athens release threshold was fitted
or evaluated. This script reads only the published aggregate
receipts; it does not load annotations or rerun model inference.

## Text for the author

> In a zero-shot pNEUMA evaluation (six recordings from four
> acquisition dates), we held the matched roster and edit count
> fixed without Athens calibration. Joint replacement improves
> on constant velocity on every date even at the lower bound
> for missing positions. A same-family control shows the same
> pattern. The joint-versus-independent envelope spans zero
> on all four dates, including when edit movement is matched
> at every forecast step.

Cite [Barmpounakis and Geroliminis](https://doi.org/10.1016/j.trc.2019.11.023)
and the [dataset release](https://doi.org/10.5281/zenodo.10491409).
The [provider terms](https://open-traffic.epfl.ch/index.php/terms-conditions/)
also request the source acknowledgment: Data source: pNEUMA -
open-traffic.epfl.ch. Neither the Word draft nor the paired v218
PDFs have been edited to incorporate this paragraph.

~~~sh
python scripts/audit_m3w_pneuma_date_robustness.py
python -m pytest -q tests/test_m3w_pneuma_date_robustness.py
~~~

[Date and leave-one-out values](readout.json).
