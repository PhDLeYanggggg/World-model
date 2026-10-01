# Decision-Aware Source Validation Mostly Abstains

## Material Passport

Fresh source-validation selection, frozen causal decisions and internal transfer
evaluation; previously trained heads are cached_verified. No new neural fitting,
independent confirmation, deployment change, Stage5C execution or SMC. See the
verification receipt for completed replay and numerical checks.

## Main Result

69/72 source fits have no supported nonempty policy under the unchanged rule.
The remaining three retain the MSE-selected head. All three are step-zero prior
decoders from the same source/context under three seeds, not three independent
datasets and not a learned cost-head improvement.

| Policy | All ADE gain vs floor | Hard gain | Mean easy gain | Intervention rate | Worst easy degradation |
|---|---:|---:|---:|---:|---:|
| MSE-selected head | +0.56809% | +0.61540% | +2.49200% | 16.77236% | 21.05476% |
| Decision-aware source choice | +0.00382% | +0.0000876% | +0.07190% | 0.26359% | 0.00000% |

At matched per-query counts the policies have identical actions, so the primary
ADE contrast and its nominal interval are exactly zero. This is no new ranking
or neural contribution, not evidence of general equivalence or useful safety.
The tiny positive aggregate gain comes from retained prior-head actions, with
almost all directions falling back to the frozen floor.

Only 9/216 transfer views have defined selected-risk denominators; 207 remain
undefined. Among those nine, all-case positive-harm ratios stay below 2%, but
easy risk fails in two views (worst 3.04230%). There are 53 selected unknown-label
occurrences. Thus the source-validation point screen does not transport as a
risk certificate even for its few surviving prior heads.

## Interpretation

The rule protects observed whole-population easy ADE mostly by abstaining.
It does not repair conditional risk learning, establish independent safety,
or improve learned decision quality over MSE selection at equal coverage.
Deployment remains unchanged. Full locality and seed summaries are in
`summary.json`; repeated windows/views are not independent samples.

Image-local detector-silver, obs8/pred12 with stride12 raw frames only. No metric,
seconds, human-gold, true3D, foundation or physical-safety claim. Independent
selection, calibration and confirmation stay closed.
