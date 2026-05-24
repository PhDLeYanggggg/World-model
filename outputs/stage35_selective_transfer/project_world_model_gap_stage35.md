# Stage35 Project World Model Gap

- Stage35 expanded external top-down trajectory coverage and made the selector conservative enough to protect easy cases.
- The strongest signal is external all-test `+12.13%` and hard/failure `+13.98%` with easy degradation `0.04%`.
- The critical blocker is t+50: Stage35 does not improve external t+50, so the cross-domain world-model candidate gate remains failed.
- Goal/interaction contribution remains weak or unproven, and latent adapters still do not provide predictive lift.
- Next shortest path: train a horizon-specific t+50 policy on the expanded external split, add stronger train-only scene/goal features, and build a real mixed SDD+external selector without damaging SDD easy cases.
