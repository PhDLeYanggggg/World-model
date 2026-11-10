# Songdo: contrast support before looking at outcomes

I checked the hashes of all 200 saved vehicle and proposal packets and
reconstructed the exact two edits for every matched arm. There are 191
eligible flights across 17 linked source components, evaluated with
three seeds. The 573 views are not 573 independent units.

| Joint versus | Different forecast paths | No shared recipient | One shared recipient | Both recipients shared | Components with any difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent | 543/573 | 248 | 290 | 35 | 17/17 |
| Additive | 543/573 | 243 | 297 | 33 | 17/17 |
| Same family | 88/573 | 16 | 47 | 510 | 16/17 |

The comparison to independent selection mostly changes recipients.
The same-family control has no waypoint difference above `1e-9` in 485
views. In 28 views it selects the same two recipients with different
family assignments. Tied views remain in the primary comparison. A
different choice does not imply a measurable gain.

## How much could the predictions differ in the score?

For waypoint predictions `p` and `q`, the absolute difference of
their errors is bounded by `||p-q||`. On a missing waypoint that
same distance is the conservative debit. For a pair of agents the
corresponding bound is the distance between their two predicted
relative positions. This gives fixed-grid bounds from the predictions
alone, with no reference labels or CV-error denominator.

| Joint versus | Median query mean bound | Median whole-flight grid mean bound | Median query pair bound |
| --- | ---: | ---: | ---: |
| Independent | 0.000616 | 0.00000927 | 0.001218 |
| Additive | 0.000615 | 0.00000939 | 0.001213 |
| Same family | 0 | 0 | 0 |

All values are in the dataset's native coordinate units. These are
upper magnitudes, not signed effects, ADE gains, confidence intervals
or risk measurements. The whole-flight column equally weights the
original queries, of which only one was edited. It does not bound an
average renormalized onto available labels or give the effect in
CV-normalized percentage points. The matched comparison and the
full-flight evaluation therefore answer different questions. No metric,
risk threshold, forecast or eligible roster was changed.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_songdo_contrast_support.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_songdo_contrast_support.py tests/test_m3w_songdo_matched_readout.py tests/test_m3w_songdo_confirmation.py tests/test_m3w_songdo_secondary.py
```

Twenty focused tests passed, including checks of the bounds under
different synthetic label masks and validation of the equal radius.
[Aggregate receipt](receipt.json). No real outcome index or coordinates,
role assignment, calibration, model selection, or deployment decision.
