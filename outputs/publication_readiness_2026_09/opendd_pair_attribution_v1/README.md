# Where the joint selection changes the forecast

I compared the frozen [imitation](../opendd_slate_shared_menu_v1/README.md)
and [direct pair-effect](../opendd_direct_pair_effect_v1/README.md) heads
against their common additive control. Both heads use the same forecast menu,
the same eight observed and twelve predicted steps, and the same number and
peak displacement of edits per query. This replay checks what changes when
the pair term is added; it does not train another selector.

| Decision change vs. additive | Imitation | Direct pair effect |
| --- | ---: | ---: |
| Changes which agents are edited | 505 | 165 |
| Same edited agents, different paths or slot assignment | 188 | 130 |
| Same assignment among the 1,067 two-edit queries | 374 | 772 |

All 1,556 one-edit queries agree with additive. The aggregate is 3,690 edits
per arm across 202 recordings, seven development locations and three seeds.
Only three locations account for the changed two-edit decisions.

## Does changing the choice help?

| Gain over additive (pp of CV error; positive is better) | Imitation | Direct pair effect |
| --- | ---: | ---: |
| Observed-step gain | +0.00420 | +0.00541 |
| Missing-step debit | 0.01423 | 0.00309 |
| Conservative gain | -0.01003 | +0.00232 |
| 95% location bootstrap of conservative gain | [-0.04651, +0.01356] | [-0.00296, +0.01039] |

At `rdb6`, changing which agents are edited accounts for -0.09500 pp of the
imitation head's -0.09615 pp conservative difference. It already loses
0.05720 pp on observed steps for those choices, before the missing-step debit.
Missing labels are therefore not the whole explanation. Both original heads
pass risk and beat both references on 34/606 views versus 32/606 for additive;
this replay does not change or reassess those risk decisions.

For relative-displacement error I separated pairs where both forecasts changed,
one changed beside an edited partner, and one changed beside an unchanged CV
partner. The three contributions use the original co-observed-step denominator
and sum to the original relative contrast up to numerical precision. All six
attribution intervals (two heads by three pair classes) span zero. There is
no clear evidence of improved relative motion, even on the pairs most
directly affected by the joint score.

## Is a better gate on these outputs worth training?

I also gave the gate the answers: for each query choose whichever of additive,
imitation and direct pair effect gives the largest gain under the same score.
This preserves each query's edit count and peak slots but uses future labels
and does not impose the recording-level 2% risk rule. The exact maximum for
this three-output menu is **+0.04417 pp** on the conservative score
(95% location bootstrap **[+0.00928, +0.08261]**), or +0.04713 pp if scored
only on observed steps. This is an upper limit on what a gate could recover
from these specific frozen outputs on this score, not a ceiling for other
forecasters and not a deployable result.

**Decision:** I would not train another gate on these three outputs. Before
fitting a new joint score, test whether a different causal candidate menu
creates material matched-budget opportunity on development sources with
co-observed trajectories. In the paper, the current pair heads are controlled
negative results, not evidence that the joint score improves forecasting.

## Reproduce

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/audit_m3w_opendd_pair_attribution.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_opendd_pair_attribution.py tests/test_m3w_pair_effect.py tests/test_m3w_slate_shared_menu.py tests/test_m3w_slate_imitation.py
```

`fresh_run`: attribution and hindsight calculation. `cached_verified`:
development cohorts, forecasts and prior readouts, verified by SHA-256,
row alignment, exact edit counts and reproduction of both old score contrasts.
The 3,000 draws resample six estimable locations after averaging their
recordings and seeds; a seventh location had no edits. The folds share
training sources, so the intervals are descriptive. No reserved outcomes,
paper files or deployment were changed. The original 2% rule is unchanged;
no Stage5C, SMC, metric, or seconds claim. [Readout](readout.json).
