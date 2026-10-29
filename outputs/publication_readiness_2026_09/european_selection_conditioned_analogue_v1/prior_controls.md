# Prior Controls and the Specific Repair

The current code and results were read before selecting this experiment.

| Existing control | Observed limitation | Why this is not a renamed repeat |
|---|---|---|
| european_easy_hurdle_v1 | Explicit occurrence supervision improves Brier but not signed risk or matched policy utility | No new occurrence/severity neural loss, weight or head is fitted. Event decomposition is diagnostic only. |
| european_selected_set_calibration_v1 | Monotone residual recalibration reduces useful coverage; source-OOF complete passes unchanged23/72 | No iterative margin, changed selected query set or transfer-calibration claim. |
| european_policy_conditional_v1 | Conditional cells gain some risk support but lose matched utility; finer selection can undo cell correction | No learned cell or threshold. Query predictions/policy are frozen. |
| european_risk_decoupling_v1 | Average fitting MSE improves while selected easy MSE does not | No optimizer or loss sweep. Same original selected population remains. |
| european_selected_analogue_v1 | Only24/72 complete donor views | Do not return to the unsupported single-source bank. |
| european_pooled_analogue_v2 | All72 supported; easy matching information present, but nearest risk magnitude overshoots and all-risk worsens | Test the mismatch between all-eligible reference labels and already-selected queries directly, keeping full-bank normalization and prediction controls exact. |

The selected reference mask comes only from frozen causal predictions. It is
not defined by observed harm/easiness or label availability. Knownness controls
supervised donor labels only. Selection conditioning may lose donor support or
discard informative cases; both outcomes remain in the primary result. It may
reduce magnitude without beating the original head. No outcome is predeclared
as success, and no new independent evidence is created by this diagnostic.

Nearest-neighbor regression and donor conditioning are conventional controls,
not a novelty claim. A useful fitting repair still needs genuinely useful
recording/scene-independent risk learning and the full paper evaluation protocol.
