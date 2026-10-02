# Failure Decomposition

## What Changed

The fitted forest predicts five conditional moments. A leaf can mix samples
with different causal disagreement envelopes. Its averaged benefit+harm may
exceed the particular query envelope. The original proportional projection
reduces both moments while leaving reference error unchanged. Algebraically,
that can change a rejected risk forecast into an eligible one.

The harm-first alternative bounds harm by the causal envelope first, then
allocates only the remaining envelope to benefit. It is a feasible readout,
not a bound on the true conditional harm. Tests establish feasibility,
monotonicity, no added actions, input immutability and unknown-label exclusion
from inference. Real-data replay establishes that the mechanical change affects
the frozen forest, not merely a synthetic example.

## Why This Is Not Enough

1. The control removes harmful occurrences, but also removes beneficial ones.
   Overall conservative utility has a nominal interval crossing zero.
2. Eight complete source passes disappear through empty action sets. No failed
   group becomes a new complete pass. Undefined risk is not zero risk.
3. The four eliminated easy-risk violations are all abstention: locality124
   controller2/head43 and locality112 controller2/heads17,29,43 now select zero.
4. The three remaining violations are locality067, controller1, heads17/29/43.
   All have nine unknown selected outcomes. Known easy harm/reference is below
   2%, but the unknown-completion envelope raises the upper to5.4486%,5.2570%
   and5.6166%. These are completion bounds, not observed error percentages.
5. The registered same-query count-matched contrast is small. It supports some
   mechanical discrimination in exposed source data, not a large learnable
   dynamics improvement or generalization claim.

## Relation to Prior Controls

Earlier neural L2 projection and mean-mass restoration also lost rare-harm
mass, and scalar restoration failed to locate harmful events. This result does
not overwrite those negatives. Its narrower new evidence is the exact current
forest decoder's effect with every tree, feature, split and threshold held
fixed. It does not justify another generic global multiplier, percentile filter,
additional calibration round or head-seed search.

## Remaining Gap

A cost estimator must distinguish conditional harm before deployment decoding,
and unknown outcomes must remain unresolved rather than being used as an
inference mask. A geometry-consistent readout alone does not establish either.
Any next fit must show a meaningful retained-utility gain under the original
risk criterion and a matched coverage comparison. Independent evidence remains
unavailable for promotion; no new claim should be selected on these source
results or on historical test outcomes.

Scope: raw-stride12 obs8/pred12, image-local detector-silver; no metric, seconds,
human-gold, physical safety, true3D or foundation claim. Stage5C/SMC off.
