# Why did the unit admission penalty barely change the policy?

## Material Passport

Fresh run: exact instrumented Torch training replay on cached-verified SOURCE fitting inputs.
No new candidate, policy score, held/calibration outcome or independent test.

All 19 complete 2,000-step replays match the frozen scientific states. PID 5503
takes 185.809s at 10971529216B peak RSS. The first full
upper control also serves as the measured local resource pilot. All original weights,
optimizer states, RNG, draws and original loss traces stay exact; added telemetry is separate.

## Activation is frequent, but gradient pressure is tiny

The 18 admitted trajectories contain 36000 updates, 4608000 repeated row draws,
174194 unknown draws and zero observed-cost supervision on unknown outcomes.
The auxiliary is positive on 33771/36000 steps (93.8083%).
Its first activation is between steps 22 and 90.
A missing/late-activation explanation is therefore not supported by these actual trajectories.

Summed auxiliary / summed anchor loss is 0.102949%.
Across 738 fixed step1/every50/final snapshots, the median auxiliary/anchor
parameter-gradient norm ratio is 0.003632%
(p90 0.021178%,
maximum 0.185241%).
The easy-channel median ratio is 0.001258%.
The total auxiliary/anchor cosine has median -0.00000999;
some gradients conflict, but conflict alone is not the measured magnitude bottleneck.

Among 2761 defined false-easy selected snapshot rows,
1789 assign below1%of total predicted harm to the easy branch.
This is consistent with a nested-output bottleneck, but does not uniquely distinguish
loss scaling, output parameterization, conditional feature quality or label noise.

## Controlled next action

Test one fixed gradient-norm balancing rule on the same model and losses. Give the
existing admission auxiliary the same pre-clipping parameter-gradient norm as the
unchanged anchor when nonzero, with a registered numerical cap. Do not change the
decoder, inputs, 2% risk rule, sampling or reserved roles. Reuse exact frozen controls
rather than claiming they were newly trained. This is an optimization repair, not
a new deferral framework or evidence that safety will pass. No threshold sweep.

The old failed utility/safety conclusions remain unchanged. These are correlated fitting
draws and training snapshots, not independent samples or population uncertainty.
No deployment, submission-ready, metric/seconds, true3D or foundation claim.
No Stage5C execution or SMC.
