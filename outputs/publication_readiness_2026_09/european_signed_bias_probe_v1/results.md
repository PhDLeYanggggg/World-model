# Fitting-Only Signed-Risk Bias Probe

## Material Passport

- fresh_run:216 exact two-axis nonnegative intercept fits;108fixedsourcegroups.
- cached_verified: existing Torch neural heads. No new neural-network training.
- Future labels used only on the two fitting sources. Held evaluation is not_run; no policy action changed.
- Already-opened development track. Independent selection, calibration and confirmation stay closed.

## Exact Fitting Result

| Parent head | Positive all/easy offsets /108 | Mean all/easy offset | Median fitting loss reduction % | Mean objective before | Mean objective after |
|---|---:|---:|---:|---:|---:|
| subset_pointwise | [92, 106] | [0.007885678688038529, 0.0034372098708300874] | 0.10341249 | 0.03196126 | 0.03191082 |
| subset_aggregate | [91, 106] | [0.0070325529907825715, 0.003366676165015653] | 0.10687642 | 0.02639945 | 0.02635670 |

Each intercept minimizes the exact source/query/subset-weighted quadratic while keeping predictions and all other parameters fixed. The score offset is constrained nonnegative. A nonpositive unconstrained optimum becomes zero.
Fitting loss improvement follows mathematically from this projected minimization: it is not evidence of downstream lift, calibration, new neural dynamics or deployment quality.
These offsets are signed-score units normalized by each fitting cost scale. They are not FDE percentages, failure probabilities, identified cost components or calibrated uncertainty.

## Seed Breakdown

| Parent arm | Forecast seed | Fitted heads | Mean all/easy offset |
|---|---:|---:|---:|
| subset_pointwise | 17 | 36 | [0.008259957972998161, 0.003400287831531952] |
| subset_pointwise | 29 | 36 | [0.008999161643726956, 0.003444478629914469] |
| subset_pointwise | 43 | 36 | [0.006397916447390465, 0.0034668631510438447] |
| subset_aggregate | 17 | 36 | [0.007148112643232669, 0.003331376865341347] |
| subset_aggregate | 29 | 36 | [0.007792142901517409, 0.0033526226266989653] |
| subset_aggregate | 43 | 36 | [0.006157403427597643, 0.003416029003006652] |

## Verification

- 24 scoped tests across3files pass.
- 1080 independent saved-parameter/quadratic checks; full108group fit replay is exact.
- The analytic loss is checked against the actual parent Torch loss, not merely an independently retyped formula. Autograd verifies its derivative on synthetic fixtures.
- Unknown fitting labels contribute zero objective weight; source and known-query balance are checked. Empty subsets contribute zero objective and zero curvature mass.
- Fresh fitting: 146.97s, peakRSS9734373376bytes, PID52193.
- Exact replay: 250.58s, peakRSS10661642240bytes, PID52492.
- Config/protocol, frozen parent checkpoint references and fit-group hashes are recorded. Large data and checkpoint files stay local.

## Next Decision

If nonnegative fitted offsets are nonzero, a separate preregistered policy experiment can test them with an identical retained-count control. If offsets vanish or are negligible, the exact objective already centers these errors and the next repair must address conditional/source-shift errors rather than a global intercept.
No held policy comparison is run in this probe. No threshold is selected from the diagnostic outcomes. The original2%screen and undefined selected-risk cases are unchanged.

## Limits

No fitting-loss percentage is a generalization percentage. Source/seed summaries repeat development contexts. No claim of independent calibration, physical safety, seconds, metric scale, human gold, true3D or foundation modeling. Obs8/pred12 raw-frame stride12 remains image-local detector-silver. Stage5C/SMC are disabled.
Full legacy integration tests and cold raw reconstruction are not_run. The research objective is not complete.
