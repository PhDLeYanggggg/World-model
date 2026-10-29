# Cyclic Complete-Region Expected-Cost Head

Research-only controller,not a new trajectory dynamics model. The trajectory
producer remains frozen.54fresh native arm64Torch heads each have26,309
parameters;all108,000primary updates completed. Three seeds refer to cost
heads,not independently retrained full forecasting pipelines.

The only primary repair changes auxiliary optimization:cycle complete fixed
old-teacher known-label recording regions,with a no-grad moment pass and a
chunked chain-rule gradient pass. Ordinary128-row base draws,initial weights,
normalizers,AdamW,2,000-step budget and positive-gain/predicted2%inference stay
matched. Added teacher-row visits total12,275,011;additional two-pass query-row
forwards total24,550,022. These are repeated fitting visits,not new data or equal
FLOPs. The loss is exact within the current region,not an all-region or
population-calibration guarantee.

Inputs remain380past/causal-rollout fields,18static region fields and7action
identities. Region and record IDs group the training loss only. No future
endpoint,central velocity,per-query future support mask,test-built goal or
test statistic enters the network. Known future costs are supervised fitting
labels only. Unknown teacher selections are not safe negatives.

Primary inner-held known ADE gain is0.943728%,with exact-minus-stochastic
change-0.000080pp,conditional95%CI[-0.108174,+0.097264]. Mixed-minus-physical
is+0.011277pp,CI[-0.121830,+0.102166]. Full-risk support24/186still fails the
required recording roster. Fewer known easy violations(89to76)do not override
that failure. The new head is not deployed and no multiplier is selected after
readout. Unknown outcomes and undefined easy risk remain explicit.

Obs8/pred12/rawstride12 EuropeanSquares SOURCE webcam detector-track pixel
coordinates,not metric/seconds,human gold,true3D or foundation evidence. These
inner-held data were already exposed development;independent roles remain
closed. Static polygons are region proxies,not verified walkable maps. Current
M3W remains a2.5D trajectory scaffold. Stage5C andSMC remain off.

CPU4/interop1/workers0,checkpoint100/heartbeat50,chunk512,10GiBdisk reserve,
12hhealthy resume boundary. Real all-bank recovery and full disabled neutral
are exact. All54endpoints freeze at8f524f40before readout. Public aggregate
checks do not reproduce raw inference without private licensed assets.
