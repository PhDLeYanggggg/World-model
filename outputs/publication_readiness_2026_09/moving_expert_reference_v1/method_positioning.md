# What This Input Contrast Can Establish

Let x be past-only history/neighbor state and a one of the existing six actions.
The frozen expert produces a complete forecast from past inputs. Its descriptor
z=phi(x,a6) is shared across all candidate actions for the same current agent
window. It is a prediction, not a future target endpoint. Learning R(x,z) and
ER(x,z) need not violate action invariance just because z describes a rollout.

The preceding model used only the first50 raw fields for its reference branch.
Those are history and neighbor motion. The remaining70 include floor/candidate
motion, rollout/history relations and path-to-neighbor-CV descriptors. Sharing
the full-amplitude version keeps them action-invariant without deleting them.
The new model does not add an image, a human goal label, new latent encoder or
an observation that was unavailable to the original forecaster.

H, EH and label-support q remain fixed per candidate. Replace conditional R/ER,
then B'=min(B,R') and compose the three changed moments with the same q. This
preserves moment coherence and changes utility only through the benefit cap.
The unchanged budget is still H-.02R<=0 and EH-.02ER<=0. Predicted feasibility
does not prove realized recording risk. A better conditional reference MSE is
not a selected-policy calibration result or a physical-safety bound.

## Related Negative Controls

- [Direct-tail pilot](../moving_direct_tail_risk_pilot_v1/README.md): directly
  learning signed quantiles did not repair the retained risk contract.
- [Selected-query critic](../european_selected_query_critic_v1/failure_analysis.md):
  lower marginal risk MSE and pooled context did not repair selected outcomes.
- [Coherent projection](../european_coherent_cost_projection_v1/readout.md):
  marginal reference error improves but the complete risk contract still fails.
- [Latest role-separated model](../moving_role_separated_cost_v1/failure_analysis.md):
  matched utility improves but selected reference/harm optimism and safety fail.

This is a specific reference-information ablation, not a claim that context
injection, conditional regression or the reused joint solver is novel. Only
the two newly matched arms isolate the context input. Comparisons with the
frozen parent additionally change reference supervision and training duration.
Neither a fitting improvement nor a development bootstrap can establish unseen
source safety or a publishable method by itself. Independent confirmation and
the original complete contract remain required for the stronger claims.
