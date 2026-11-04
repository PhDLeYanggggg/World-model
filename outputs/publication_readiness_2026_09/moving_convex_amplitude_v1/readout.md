# Matched Convex Amplitude Decoder

## Material Passport

Fresh_run: 18 real Torch convex heads, 36,000 new primary updates, complete
first-convex refit and first-control refit, new held convex decisions.
Cached_verified: all 18 original unconstrained control heads and their
saved decisions, with fresh exact fitting/held predictions and policy
scalar replay. This is not 36 fresh fits. Separate full replay verifies
all new outputs/scalars and saved solver feasibility, not bitwise MILP
reoptimization. Original 24-group primary remains not estimable.

## Fixed Supported-Group Development Contrast

Convex minus unconstrained joint-free known: -0.0219 [-0.5299,+0.5058] pp.
Paired completion lower: -1.8308 [-3.6239,-0.4230] pp.

All intervals use 3,000 source-group resamples after producer/actual-seed
averaging. They are exposed-development intervals, not independent final
confirmation or corrections for the research search. No held threshold,
seed, checkpoint or scientific endpoint was selected.

| Arm | Policy | Known gain over floor (%) [95% CI] | Completion lower (%) [95% CI] | Original support | Known all/easy violations | Undefined easy | Worst known easy-net (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| convex | floor | +0.0000 [+0.0000,+0.0000] | +0.0000 [+0.0000,+0.0000] | 0/348 | 0/0 | 348 | 0.0000 |
| convex | frame_uniform | +5.8844 [+2.7519,+9.6416] | +2.0664 [-2.9822,+6.1949] | 1/348 | 144/162 | 165 | 290.7071 |
| convex | independent | +10.3883 [+4.9014,+16.1256] | +5.7343 [-2.7029,+12.8759] | 10/348 | 267/275 | 48 | 428.9130 |
| convex | joint_free | +11.7058 [+5.4678,+18.2021] | +5.4918 [-3.2777,+12.9002] | 10/348 | 268/278 | 47 | 428.9130 |
| convex | joint_matched | +11.5472 [+5.3731,+17.9919] | +5.4246 [-3.3120,+12.7749] | 9/348 | 269/275 | 50 | 428.9130 |
| convex | uncontrolled_neural | +16.4074 [+7.4982,+25.5310] | -5.0187 [-16.4749,+5.0800] | 6/348 | 304/320 | 15 | 1742.9603 |
| unconstrained | floor | +0.0000 [+0.0000,+0.0000] | +0.0000 [+0.0000,+0.0000] | 0/348 | 0/0 | 348 | 0.0000 |
| unconstrained | frame_uniform | +6.2459 [+2.8258,+10.3041] | +2.2958 [-2.4810,+6.3846] | 1/348 | 146/166 | 160 | 305.3475 |
| unconstrained | independent | +10.6394 [+4.7628,+16.7147] | +5.4507 [-3.4143,+12.9332] | 12/348 | 270/276 | 47 | 376.3311 |
| unconstrained | joint_free | +11.7276 [+5.1202,+18.6581] | +4.5682 [-4.4820,+12.0322] | 10/348 | 273/281 | 46 | 376.3311 |
| unconstrained | joint_matched | +11.6428 [+5.1058,+18.4680] | +4.5969 [-4.4231,+12.0125] | 7/348 | 276/281 | 50 | 376.3311 |
| unconstrained | uncontrolled_neural | +16.4074 [+7.4982,+25.5310] | -5.0187 [-16.4749,+5.0800] | 6/348 | 304/320 | 15 | 1742.9603 |

The floor makes no path change; undefined selected-risk support is not
evidence that it causes harm, nor an empirical finite-support certificate.
Every unsupported/empty-event view is retained. All known easy/hard,
available-label ADE, mean group-p95 and available-endpoint FDE diagnostics
are in `secondary_metrics.json`. Missing required subsets stay not
estimable; endpoints/partial labels are not a full-roster 12-step claim.

## Intervention Size And Fitting Diagnostics

| Arm | Source-macro switch (%) | Source-macro mean amplitude, all rows | Empty same-count group views |
|---|---:|---:|---:|
| unconstrained | 70.9258 | 0.658174 | 3/138 |
| convex | 69.5677 | 0.629830 | 3/138 |

Same-count comparisons are utility only. Empty is not estimable, not a
safe zero; equal counts need not mean equal path-change amplitude.
Histograms count repeated row views, unlike source-macro switch means.
Own fitting-region known H / predicted H: 40.662441259599824; EH / predicted EH: 245.05945347218565.
All 18 constrained H/EH fitting-shape checks zero: True.
These are pooled resubstitution cost masses, not calibration or a causal
decomposition. Correct shape need not mean correct cost magnitude.
The fitting receipt distinguishes zero prediction with positive actual
harm from both masses zero; neither is relabeled as safe zero.

## Solver And Execution

| Arm | Repeated query views | Nonoptimal matched | Nonoptimal free |
|---|---:|---:|---:|
| unconstrained | 16734 | 323 | 175 |
| convex | 16734 | 374 | 173 |

pilot: 58.453 s; peak RSS 9,306,439,680 bytes.
train: 1627.621 s; peak RSS 9,338,159,104 bytes.
evaluate: 841.278 s; peak RSS 5,691,883,520 bytes.
verify: 180.764 s; peak RSS 5,530,927,104 bytes.

Fixed 0.25s/query; timeout/post-check failure retains the original
predicted-feasible independent parent, not a true-risk pass or exact
optimum. Joint allocation pools predicted budgets, not collision costs.
Code/config/normalization/input/coverage/budgets are frozen. The decoder
uses nonnegative convex hinges with stable linear arithmetic, not a
precision ablation or an invented general convex-network architecture.
FP32 fitting loss and FP64 hinge inference are disclosed in the protocol.
No independent calibration or confirmation was accessed. This is not
new trajectory dynamics, JEPA, latent generation or deployment. Pixels
and native annotation steps are not verified metres/seconds or SDD t50.
Partial-label error and whole-missing bounds leave partial steps unknown.
No true3D, foundation, physical-safety or submission-ready claim.
Stage5C execution and SMC remain disabled.

The complete source breakdown is in `per_source.md`; no favorable
source subset is promoted to a new primary. Full error/subset ledgers,
solver outcomes and the negative decision remain available.
