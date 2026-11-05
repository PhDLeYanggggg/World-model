# Channel-Specific Risk Heads: No Safe Decision Repair

I separated the all-risk and easy-risk regression heads while keeping the
original targets, causal features, weights and decision threshold. The primary
uses two 32-tree forests against one shared 64-tree forest. This changes the
model, not the original 2% evaluation contract.

| Context | Model | MSE reduction % | Decision-cost reduction % | Accepted | Harmful |
|---|---|---:|---:|---:|---:|
| critic0 | shared64 | +0.000000 | +0.000000 | 1 | 0 |
| critic0 | shared32 | +0.863603 | +0.066610 | 3 | 0 |
| critic0 | separate64 | +2.908763 | -0.033689 | 0 | 0 |
| critic1 | shared64 | +0.000000 | +0.000000 | 10 | 1 |
| critic1 | shared32 | +0.403959 | -0.001182 | 17 | 4 |
| critic1 | separate64 | +0.372358 | +0.461475 | 113 | 19 |
| critic2 | shared64 | +0.000000 | +0.000000 | 0 | 0 |
| critic2 | shared32 | -0.991956 | +0.000000 | 0 | 0 |
| critic2 | separate64 | -3.898268 | +0.000000 | 0 | 0 |

Positive reductions are better. The primary information screen passes
**0/3** contexts; two were required. No deployment promotion.
These are source-macro known-proposal decision costs, not trajectory ADE
improvement, independent test results or a full-policy safety certificate.

## What This Rejects

Removing shared partitions is not sufficient to repair safe admission.
Context1 switches more often but harmful admissions increase from 1 to 19.
All 113 primary admissions come from source076; their descriptive normalized
known easy-harm/reference ratio is 5.820542%. The other sources admit none,
which leaves empty support NA rather than safe. Context2 still rejects all
38,579 known proposals and has worse MSE. The secondary shared32 control
cannot retrospectively replace the primary and does not satisfy the screen.

The earlier ego-frame improvement occurred almost entirely in unchanged
rejects. That boundary finding is preserved in the separate audit. Lower
regression loss and more admissions are each insufficient outcomes.

## Scope And Checks

- fresh_run: 16 new model objects containing 24 forests, full refits and boundary attribution.
- cached_verified: eight shared64 controls, original data/schema/ID bindings and producer lineage.
- not_run: new neural dynamics, full-policy risk, nested-C/H or independent confirmation.
- All 24 comparison cells refit exactly; shared32 equals the shared64 first-tree prefix.
- Native arm64, CPU4, inter-op1, workers0; sklearn forest n_jobs4.
- The reused runtime banner still says forest_threads1; actual registered forest setting is4.
- Partial-label supervision and dependent windows remain; eight sources are not eight independent seeds.
- Tree counts are matched, not nodes/parameters; every node count is retained in the JSON.
- Original 2% all/easy harm, easy-net and support rules remain. No identity exemption.
- Original24-group primary remains NA. Reserved Moving20, DUT, DroneCrowd and H stay closed.
- Native obs8/pred12 silver image coordinates, not metric, seconds, true3D or foundation.
- Stage5C/SMC disabled. No submission-ready claim.

## Fitting Versus Source-Held Diagnostics

The following posthoc comparison reuses completed models. It is not model
selection or a new split. Training performance is optimistic by construction.

| Excluded source | Shared64 fit cost | Shared64 held cost | Separate64 fit cost | Separate64 held cost |
|---|---:|---:|---:|---:|
| moving_source_component_048 | 0.0041702949 | 0.0025842651 | 0.0040414046 | 0.0025865721 |
| moving_source_component_067 | 0.0025865721 | 0.0042636851 | 0.0025843322 | 0.0042636851 |
| moving_source_component_053 | 0.00080558852 | 0.0015451593 | 0.00062850189 | 0.0015451593 |
| moving_source_component_065 | 0.0011186662 | 0.0009190038 | 0.0011186662 | 0.0009190038 |
| moving_source_component_076 | 0.0011461938 | 0.00069151131 | 0.0010995047 | 0.00067694866 |
| moving_source_component_059 | 0.0018930968 | 0.011806774 | 0.0018930968 | 0.011806774 |
| moving_source_component_060 | 0.005824189 | 0.0023488942 | 0.0066220367 | 0.0023488942 |
| moving_source_component_073 | 0.0064294699 | 0.0014372994 | 0.0070778341 | 0.0014372994 |

[Protocol](protocol.md) | [Failure analysis](failure_analysis.md) | [Reproduction](operation_zh.md)
