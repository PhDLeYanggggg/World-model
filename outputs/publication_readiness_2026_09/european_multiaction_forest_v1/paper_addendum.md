# A Conventional Estimator Control On The Current Action Bank

Working evidence addendum, 29 October 2026. This supplements, rather than
rewrites, the frozen manuscript and earlier negative experiments. It is not
a submission-ready paper or independent confirmation result.

## Design

We fit 54 ExtraTrees continuous-cost regressors: six SOURCE contexts, three
controller seeds and physical-only, physical+neural and neural-only banks.
Each uses 128 trees, maximum depth 12, minimum leaf size 64 and one third of
features per split. These settings precede the new readout. The older binary
forest has incompatible roles and action inputs and is retained only as a
historical negative reference.

Each estimator receives the same 398 causal/static fields, action identity and
causal envelope information as the frozen static-scene neural comparator.
Physical-fitting normalization is shared. We replay the original 2,000 x 128
source-balanced query draws and fit the weighted empirical query/action
measure. No future outcome, future mask, central velocity or test-endpoint
goal is an inference input.

The forest minimizes symmetric five-moment squared error and applies legal
moment projection afterward. The neural comparator uses asymmetric harm loss,
ranking and a constrained decoder. Therefore this is an estimator/objective
control, not a matched-FLOP or single-factor architecture ablation. No new
forecasting producer is trained. All endpoints freeze at commit `da1684d1`
before the first readout.

## Results

| Candidate bank | Forest known ADE gain over floor (%) | Neural cost-head gain (%) | Paired forest minus neural (pp), conditional 95% CI | Required risk support |
|---|---:|---:|---|---:|
| Physical | -0.000083 | 1.076598 | -1.076681 [-2.523074,-0.258453] | 120/186 |
| Physical + neural | -0.000073 | 0.982769 | -0.982843 [-2.301874,-0.242325] | 118/186 |
| Neural only | 0.044378 | 0.433381 | -0.389003 [-0.666399,-0.152410] | 65/186 |

The mixed forest's count-matched change is -0.009443pp, CI
[-0.023258,+0.000643]. Its neural-candidate increment over its physical-only
bank is +0.0000094pp, CI [-0.0000240,+0.0000517], not a positive contribution.
No seed, bank or fixed 2x/4x harm sensitivity is selected after readout.

Interventions decrease from 32,219 to 3,457 repeated held row views. Although
record-risk support rises from 23/186 to 118/186, 66 easy-risk views remain
undefined, two have known easy-risk violations and 43 selected outcomes are
unknown. The complete 2% all/easy contract is not supported. Empty risk is
not_estimable; count matching measures utility only. The completion-lower
forest-minus-neural contrast is -3.200105pp, CI [-8.549639,-0.355064]. This
normalizes utility differences by fixed known floor-error mass, not an
identified fully observed population-error denominator.

## Fitting-Region Mechanism

We replay 1,332 paired fitting record/head/bank views, with no optimization.
Both estimators are scored on identical frozen action regions. In all-legal
mixed-bank pools, median predicted/observed EH is 1.016480 for forests and
1.026498 for neural heads. In the frozen neural-selected regions it is
4.576410 versus 0.021367, using the same 341 positive-denominator and 103
undefined record views. Thus near-unit all-legal aggregate ratios do not
establish accurate costs in the region used for decisions. This measured
contrast does not uniquely identify architecture as its cause.

The experiment rejects this fixed conventional control as a repair. It does
not establish that every conventional estimator fails or that the current
features are information-theoretically insufficient. Bank-specific gate
suppression and conditional risk estimation remain testable explanations.

## Reproducibility And Limits

Primary fitting takes 2,454.887 seconds on native arm64 CPU4, with peak RSS
10.74 GB and 335.4 MB of private forest checkpoints including engineering
refits. The first 128-tree refit and all 144 readout groups replay exactly.
The 1,955,179-byte aggregate metric archive is lossless and contains neither
raw data nor model weights. The separate fitting diagnosis also replays exactly.

Intervals use 3,000 SOURCE locality resamples after averaging repeated context
and controller-seed views. The 186 held recording views repeat 31 unique
recordings. They are exposed development, not independent confirmation,
whole-pipeline OOF or three independently trained forecasting pipelines.
Data are oblique webcam detector-track box midpoints in pixels, with eight
observed and twelve predicted steps at twelve raw frames per step. They are
not SDD human gold, metric/seconds, true 3D or foundation-model evidence.
Independent roles and DroneCrowd remain closed. Deployment, Stage5C execution
and SMC do not change.

## References And Artifacts

- Geurts, Ernst and Wehenkel. [Extremely randomized trees](https://link.springer.com/article/10.1007/s10994-006-6226-1). Machine Learning 63, 3-42, 2006.
- [scikit-learn 1.8.0 ExtraTreesRegressor](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html).
- [Fixed protocol](protocol.md), [method details](method_note.md), [verified results](readout.md), [failure analysis](failure_analysis.md), [Chinese reproduction guide](reproduction_zh.md).
- [Generated statistical comparison](figures/estimator_comparison.svg). This figure contains aggregate experimental statistics, not dataset imagery.
