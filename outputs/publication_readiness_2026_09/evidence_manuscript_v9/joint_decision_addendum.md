# Joint-Decision Loss: A Negative Safety Follow-Up

This addendum is separate from the compiled main PDF. It does not upgrade
the paper's main-method, independent-confirmation or submission-ready claims.

## Hypothesis And Controlled Test

Zero-target channels can veto a proposal while receiving no signed-cost
gradient. I compared the preceding per-channel auxiliary loss with a loss on
the complete two-head, two-channel conjunction. All six 64,260-parameter
models use identical warm starts, source-excluded proposals, preprocessing,
query draws and 2,000-update budgets. Original conservative unknown labels,
2% safety limits, support conditions and thresholds remain unchanged.

The three fresh controls exactly reproduce their predecessors. Both recovery
pilots, the first complete treatment refit, every fitting attribution and all
held predictions/actions/metrics replay exactly. These checks establish
same-data reproducibility, not independent scientific confirmation.

## Results

Joint fitting costs fall 13.90%, 17.58% and 30.11%. Zero-only rejection costs
fall 45.32%, 52.55% and 67.36%. Harmful admissions nevertheless increase:
false-safe counts change from 16/20/22 to 18/35/162 across the three contexts.

| Policy | Source-macro known ADE gain % | Hard gain % | Worst defined easy-net degradation % | Full original risk/support |
|---|---:|---:|---:|---:|
| Per-channel control | 0.758534 | 1.008910 | 8.037815 | 0/17 |
| Joint-decision model | 0.894772 | 1.169812 | 8.633364 | 0/17 |

The matched known gain is +0.136238 pp with missing-outcome completion bounds
[+0.092671, +0.179805] pp. These are not sampling confidence intervals. The
seven SOURCE units / 17 recording views were already exposed development
data with one producer seed. SOURCE075 has a negative contrast. The joint
policy has 13 known all-risk and 13 known easy-risk violations, plus three
undefined selected-easy views. No policy is promoted.

## Interpretation And Limitations

The loss repairs a measured fitting omission but does not solve safe transfer.
Lower pooled decision cost can coexist with more harmful admissions and worse
recording-level easy degradation. Removing the decision head increases known
gain but produces 40.38% worst easy-net degradation, not a valid safety repair.
No new inference threshold or identity-fallback certification was introduced.

An earlier all-action-averaging explanation is explicitly retracted: each
supported row already supervises one frozen proposal, and opposing all/easy
target signs are absent in observed fitting proposals. The present loss also
changes gradient routing/scale; those effects are not separately identified.

Missing support leaves the original 24-group primary NA, not passed. No new
reserved/H evidence, multi-seed claim or forecasting improvement is supplied
by these risk-head fits. Native obs8/pred12 image coordinates only, not metric,
seconds, true 3D or foundation modeling. Stage5C and SMC remain disabled.

[Full experiment](../moving_joint_decision_v1/README.md) |
[Failure analysis](../moving_joint_decision_v1/failure_analysis.md) |
[Corrected diagnosis](../moving_decision_semantics_v1/README.md)
