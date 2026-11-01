# Reproduction Scopes

Two complementary checks are kept distinct. Neither changes the registered
scores,primary endpoint,cohort or risk gate.

1. `run_m3w_pair_action_allocation.py verify` reruns inference,the timed MILPs
   and complete scoring. It requires exact equality to every original group.
   A two-second wall-clock cutoff may change which queries finish. If this
   check fails,the failure must remain visible;it is not silently renamed
   an exact optimization replay.
2. `verify_m3w_pair_frozen_actions.py` reruns all original predictor inputs and
   verifies complete action-cache bindings. It freshly recomputes every causal
   graph,acceptance objective,original budget/count check and full outcome score
   using the frozen chosen actions. It does not run the timed solver again.
   Its receipt explicitly sets optimization_replayed=false and cannot replace
   a failed fresh-optimization reproducibility check.

The second check is implemented before the complete result is inspected. It
supports faithful reproduction of an archived finite-budget decision,not a
deployment policy that memorizes exposed SOURCE outcomes. Timed solver failures
in the original run retain the unary parent and stay in all population scores.

No whole-pipeline OOF,independent confirmation,new training,physical safety or
venue-readiness claim follows from either engineering check alone.
