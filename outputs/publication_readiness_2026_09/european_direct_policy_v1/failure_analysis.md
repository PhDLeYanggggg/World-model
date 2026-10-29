# Failure Taxonomy: Direct Neural Intervention

## Confirmed Findings

1. Benefit removal, not only reduced harm. Held known benefit falls5.052642->
   1.055998pp of the full known reference; known harm falls1.453703->0.291517pp.
   The known net difference is-2.834457pp. These are equal-view normalized mass
   components, not pooled raw coordinates. The saved decomposition replays.
2. Conditional choice quality remains weak. Same-count utility versus original
   stays entirely negative. Thus intervention-count reduction alone cannot
   explain the loss.15 empty matched views remain, with risk not estimable.
3. Recording transfer fails. Primary known easy violations rise5->50 from fit to
   held. Full-policy budgets fall35/72->4/72. The three head-seed held utility
   upper bounds are all negative, not a single-seed sign reversal.
4. Missing outcomes are not the whole cause. Held known-label violations are
   all43/easy50. Additional unknown-only upper failures are9/6. These categories
   overlap across channels;11 easy ratios are undefined and do not pass.
5. Empirical constraints are not fully optimized. Primary has28 fit known
   all-risk and5 easy-risk violations. The final dual snapshot has30/888 entries
   at cap100, but this does not prove that the cap caused the violated records.
   Final minibatch losses are not whole-recording constraints.
6. Hard forward actions are not a sufficient remedy. Hard-versus-soft-record
   matched utility improves slightly, but full utility is unresolved and known
   easy-risk violations worsen39->50 on held recordings. Per-record duals alone
   also do not establish a held improvement over source-level duals.

## Hypotheses Still Unresolved

- Rare-harm minibatch coverage versus inadequate full-record constraint updates.
- Finite dual cap, optimizer timescale and annealed surrogate gradient effects.
- Gain/harm information missing from the fixed causal features.
- Producer/recording distribution differences and repeated historical exposure.

These explanations are not separately identified by this experiment. Switching
from the earlier moment head to the new direct head changes several components;
only the two within-experiment contrasts isolate constraint grouping and hard
forward decisions. Do not label all differences as an effect of removing MSE.

## What Was Repaired Successfully

The first real pilot stopped before training because a new verifier hashed arrays
with a different codec than the old receipt writer. Identical arrays reproduced
the mismatch. Execution revision2 uses the original codec; the numerical model,
features, splits, losses and thresholds are unchanged. The original registration
and failed pilot are retained. Repaired real-data resume is exactly deterministic.
This is an implementation repair, not the reason for the scientific failure.

## Shortest Next Step

Measure exact full-recording training constraint residuals and gradient coverage
before changing model capacity. Compare a whole-recording constraint-update path
with the current stochastic path, fixed predictive inputs and outcome budget.
Require actual useful full-policy decisions; no new risk-MSE proxy gate, empty
control safety pass, posthoc threshold rescue or better-secondary-arm promotion.

The data remain original-TRAIN exploratory. Independent roles remain untouched.
No new causal trajectory producer was trained. No metric/seconds,human-gold,
true3D,foundation,physical-safety or submission-readiness claim. Stage5C/SMC off.
