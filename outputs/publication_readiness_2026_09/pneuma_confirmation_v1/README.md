# Athens: four-date zero-shot comparison

All six frozen members from four acquisition dates are confirmation
members. The author selected the all-date zero-shot comparison
before any real future positions were opened. Athens provides no
calibration data or release threshold. The analysis uses the
original 8/12 native grid and the prior-only donor cutoffs.

The [registration](registration.json) binds the complete original
matched roster, forecast bank, recipient/path attribution and
stepwise movement controls. The [freeze](freeze.json) fixes the
operating rule before scoring. The published registration and freeze
must match the local files before a target can be joined. The main
comparison estimates utility on this single-city cohort; it does not
authorize a deployment decision.

## Scored result

The frozen registration was published in `84b9e944e` before the
confirmed readout. The 643 matched queries contain 67,962 actor
histories, including 63,346 with complete futures. There are 1,929
query-seed views and two edits per query in each proposal arm.
The three seeds do not increase the number of independent dates.

| Joint versus | Observed gain (pp) | Missing-position envelope (pp) | 95% date interval for the lower endpoint |
| --- | ---: | ---: | ---: |
| CV | +0.05621 | [+0.02383, +0.08858] | [+0.00622, +0.04144] |
| Independent | +0.00439 | [-0.04427, +0.05304] | [-0.05378, -0.03126] |
| Additive actor head | +0.00407 | [-0.04448, +0.05263] | [-0.05307, -0.03078] |
| Same family | -0.00084 | [-0.00156, -0.00012] | [-0.00323, -0.00005] |

The small gain over the unedited CV reference survives the
missing-position bound on this sample. It does not establish a joint
advantage over independent or additive selection at equal spend.
Against independent selection, the relative-position envelope is
[-0.00225, +0.00290] in release coordinates. Matching movement at
every step gives an individual-utility envelope of
[-0.03823, +0.04511] pp. In the crossed analysis the path term is
negative in the sample envelope [-0.00246, -0.00010] pp, whereas
the recipient term's sign is unresolved. These are prespecified
secondary comparisons, not extra independent samples.

| Forecast | Available ADE | Available FDE | Available ADE p95 |
| --- | ---: | ---: | ---: |
| CV | 2.09723 | 5.07849 | 7.07755 |
| OLS4 | 2.44369 | 5.65265 | 8.28060 |
| Independent | 2.09597 | 5.07829 | 7.07401 |
| Additive actor head | 2.09595 | 5.07824 | 7.07407 |
| Same family | 2.09597 | 5.07741 | 7.07221 |
| Joint | 2.09598 | 5.07743 | 7.07221 |

Errors use native release coordinates. Each proposal arm changes
2.85% of actors on the date-averaged roster. Joint passes the original
risk-and-dual-utility check in 1 of 18 member-seed views; these are
fixed proposals, not releases. Its observed signed easy and hard
gains versus CV are +0.050% and +0.128%. Neither figure replaces the
risk rule. There was no Athens calibration or deployment test.

Intervals use 3,000 descriptive resamples of four acquisition dates
after seed and member averaging. The four dates are from one city.
The measured scope is therefore the approved zero-shot utility
comparison, not multi-city generalization.

[Primary readout](confirmation.json) |
[Recipient/path attribution](recipient_path_attribution.json) |
[Stepwise sensitivity](stepwise_budget_sensitivity.json).
