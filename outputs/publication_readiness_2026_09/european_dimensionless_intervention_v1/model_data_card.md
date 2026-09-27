# Model and Data Card

Frozen dimensionless forecast bank, plus separate three-moment envelope heads
for it and fixed damping. Heads have 22,914 parameters each, width 64, GELU, and 2,000 updates.
Source-only controller preprocessing; twelve opened source localities, three
producer groups, six ordered role assignments, seeds 17/29/43. No readout fitting.
All 108 heads are fresh training; forecasters are hash-verified cached results.

The 355 causal features contain target/neighbor histories, full predicted
candidate/CV rollouts and observed scale. No future endpoint, mask, target latent,
central velocity, test endpoint goal or test-fitted statistics. Label-derived
hard/easy events are evaluation/training labels, not inference features.

Fixed 2% predicted all/easy moment screens, no readout threshold tuning. Their
predicted ratios are not realized guarantees. Joint controls optimize an image
proximity proxy at a matched count; it is not a physical collision measure.
Nonadditive geometry support and numerical solver failures must accompany claims.

318,969 source-training target histories; recordings and localities are not new
confirmation datasets. Silver detector tracks, image-local coordinates,
obs8/pred12 raw stride 12. Independent roles remain closed. No deployment,
Stage5C, SMC, metric/seconds, true3D, human-gold or foundation claim.
