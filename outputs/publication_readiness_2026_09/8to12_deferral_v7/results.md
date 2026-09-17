# v7 Cost-Sensitive Deferral: Complete Registered Control

All 24 new heads completed 1,000 updates. Existing forecasting checkpoints are hash-verified, not retrained.
Each head uses the same 11,966 OOF queries and 306 causal features as the original gain/harm heads.
Fresh floor/candidate scoring exactly matches all parent query exports before ordinary controls are reused.

Primary: eight observed / twelve predicted native annotation steps, past-normalized ADE, equal physical scene.
Dataset-local geometry is unverified. Neither seconds nor metres are claimed.
Only one exposed physical development site is available. Three training seeds are not three independent sites;
scene CI is not estimable here. No threshold tuning, best-deferrer selection or deployment occurs.

The unconstrained two-action loss follows the single-expert adaptation of
[Mao et al., ICML 2024](https://proceedings.mlr.press/v235/mao24d.html).
This is not a reproduction of their experimental results or a calibrated risk guarantee.

## Every Registered Setting

| Predictor | Deferrer | Mean ADE gain vs CV | Seed SD (pp) | Positive seeds | Easy <=2% seeds | Both |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| skip | linear_bound1 | -0.338658% | 0.350279 | 0/3 | 0/3 | 0/3 |
| skip | linear_bound10 | -0.257130% | 0.258356 | 0/3 | 0/3 | 0/3 |
| skip | mlp_bound1 | -0.131560% | 0.184547 | 0/3 | 0/3 | 0/3 |
| skip | mlp_bound10 | -0.194260% | 0.066447 | 0/3 | 0/3 | 0/3 |
| bounded | linear_bound1 | +0.066136% | 0.001656 | 3/3 | 0/3 | 0/3 |
| bounded | linear_bound10 | +0.073370% | 0.010242 | 3/3 | 0/3 | 0/3 |
| bounded | mlp_bound1 | +0.049541% | 0.026322 | 3/3 | 0/3 | 0/3 |
| bounded | mlp_bound10 | +0.058312% | 0.005992 | 3/3 | 0/3 | 0/3 |

## All Seed Results

| Predictor | Seed | Deferrer | Gain | Hard gain | Easy degradation | Switch rate |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| skip | 17 | linear_bound1 | -0.010346% | +0.002313% | 162.0119% | 9.490% |
| skip | 17 | linear_bound10 | -0.037732% | -0.000269% | 473.8102% | 14.912% |
| skip | 17 | mlp_bound1 | -0.034211% | +0.008467% | 532.2267% | 18.216% |
| skip | 17 | mlp_bound10 | -0.246177% | -0.022344% | 2743.6929% | 15.579% |
| skip | 29 | linear_bound1 | -0.707395% | -0.064179% | 7868.8260% | 12.394% |
| skip | 29 | linear_bound10 | -0.541890% | -0.011229% | 6536.9489% | 56.318% |
| skip | 29 | mlp_bound1 | -0.344398% | -0.030551% | 3839.7724% | 7.701% |
| skip | 29 | mlp_bound10 | -0.217228% | -0.023580% | 2372.0120% | 8.564% |
| skip | 43 | linear_bound1 | -0.298232% | -0.017379% | 3436.3683% | 7.283% |
| skip | 43 | linear_bound10 | -0.191767% | -0.006063% | 2280.9767% | 13.588% |
| skip | 43 | mlp_bound1 | -0.016070% | +0.008906% | 314.2853% | 20.146% |
| skip | 43 | mlp_bound10 | -0.119376% | -0.002156% | 1437.9237% | 10.298% |
| bounded | 17 | linear_bound1 | +0.067950% | +0.064090% | 32.9067% | 86.856% |
| bounded | 17 | linear_bound10 | +0.071406% | +0.070133% | 42.3674% | 85.265% |
| bounded | 17 | mlp_bound1 | +0.070408% | +0.070890% | 53.8671% | 81.575% |
| bounded | 17 | mlp_bound10 | +0.064106% | +0.061342% | 31.3867% | 57.287% |
| bounded | 29 | linear_bound1 | +0.064705% | +0.063334% | 51.6294% | 57.006% |
| bounded | 29 | linear_bound10 | +0.084452% | +0.092577% | 77.9173% | 75.349% |
| bounded | 29 | mlp_bound1 | +0.019969% | +0.020496% | 14.3981% | 10.494% |
| bounded | 29 | mlp_bound10 | +0.052139% | +0.051085% | 28.3708% | 28.474% |
| bounded | 43 | linear_bound1 | +0.065754% | +0.067229% | 63.3004% | 83.163% |
| bounded | 43 | linear_bound10 | +0.064252% | +0.068704% | 78.2604% | 81.702% |
| bounded | 43 | mlp_bound1 | +0.058244% | +0.066034% | 96.9598% | 72.842% |
| bounded | 43 | mlp_bound10 | +0.058689% | +0.062623% | 68.7087% | 68.966% |

## Scope of the Comparison

All four original cost-head/policy settings are retained in metrics.json, with 480 paired deferrer/control
comparisons. They share forecasters, OOF inputs and query populations, not realized intervention counts
or imposed risk budgets. The M3W guards are absent from deferral by design. Lower guarded error alone
therefore cannot establish a superior learning objective; coverage/budget is a confound.
The previously completed actual-count-matched joint/unary experiment still shows no joint advantage.

Cost clipping changes the fit objective, not evaluation. Per-head clip/tie fractions and first/last-50
minibatch-loss means are in metrics.json; loss reduction does not prove convergence or forecasting lift.
All per-recording normalized metrics, absolute easy error, positive harm, FDE and switch denominators remain available.
Raw50 is the completed parent supplementary task and is not re-evaluated for these new deferrers.

Stage5C and SMC remain disabled. No new deployment or submission-ready claim follows from this comparator.
