# Avoiding A Renamed Repeat Of Existing Risk Objectives

## Evidence Role

Fresh static inspection of existing tracked code and result documents on
29 October 2026. Listed files have no difference from the current committed
version; SHA256 bindings are below. This verifies document/code identity,
not a fresh rerun of the older numerical experiments, their private caches
or their no-leakage pipelines. No older held role is reopened here.

## Existing Implementations That Already Address Nearby Ideas

| Proposed shortcut | Existing implementation | Consequence for the next experiment |
|---|---|---|
| Regress harm minus the 2% reference budget directly | `src/world_model/m3w_risk_excess.py` | Signed-budget supervision already exists; it is not a new method merely because five outputs are replaced by a margin. |
| Add utility and all/easy empirical risk terms | `m3w_task_risk_training.py` and `m3w_task_risk_objective.py` | Existing differentiable action regret and bounded dual updates already do this. A new seven-action completion version would need an explicit additional hypothesis. |
| Supervise easy occurrence and conditional easy harm separately | `m3w_easy_hurdle.py` | Existing supervised factorization separates occurrence, reference and harm. Better probability loss is not evidence of safe selected-risk control. |
| Reweight the model's currently selected region while training | `m3w_adaptive_region_cost_head.py` | Existing fit-only refreshed selection emphasis already uses causal predictions. Renaming it on-policy training does not establish novelty. |
| Replace individual losses with query-aggregate signed risk | Existing `european_query_excess_refit_v1` conclusion | The historical comparison is negative/incomplete, not an unused obvious repair. Its different cohort and action task must remain explicit. |

The older reports record useful local or loss-level changes as well as failed
complete safety/development checks. They are not directly pooled with this
72-head study: action banks, roles, cohorts and output parameterizations
differ. No historical held result becomes independent again by reusing it.

## Decision

Do not launch another broad margin/dual/reweighting grid as if those mechanisms
had never been tried. The current controlled result adds a different fact:
valid unknown-row endpoints can improve fit on the old region yet induce a
new region with severe risk underprediction. A pure fixed-head availability
veto is additionally ruled out as a complete-support repair for this cohort.

A next training intervention must state what changes conditional prediction
on newly admitted, causally identifiable regions, and distinguish it from
these existing controls. Alternatively, add admitted predictive information
or lawful labels before retraining. Matched SOURCE RGB remains unavailable
in the audited local release; a present-day webcam snapshot is not a matched
historical frame. None of these observations justifies changing the 2% gate,
using future-label support in inference, opening independent roles, or
claiming an information-theoretic impossibility for the broader project.

## Code Bindings

```text
2c6260454ed3f30c743a4e74ee9f7196da3975c84407cc5f38a4054532e67088  src/world_model/m3w_risk_excess.py
a91f1a9bb4b1c097c77f692f35a8787170095fab10b7c95bed8167ab7d92e3d6  src/world_model/m3w_task_risk_training.py
a93ecc7d804260b40243c86bf87772fec983af2b59f8a59bd777100233892043  src/world_model/m3w_task_risk_objective.py
827c6a96052254fcba96a622a72d9474c76fc43cb6d71da400fc3faa39406586  src/world_model/m3w_easy_hurdle.py
fefb1f1dfe96eeebdafdcee769fdfd8a240e85ed145d52ca65424113e86e6b2f  src/world_model/m3w_adaptive_region_cost_head.py
```

## Result-Document Bindings

```text
16cd511d8cbe83d4fe7659943a4581ea488c779655da8edc32053d08ed9eaa22  european_risk_excess_v1/results.md
17b5ba30cba97738ecde93ad8c5b23cd506f77a03c114f4b3c8d53f7fb5d4aaf  european_task_risk_v1/readout/conclusions.md
25d3b7cb7d952f19a37c8d886d1abb67878e29cb0a173df1a02b2e4d9c597833  european_query_excess_refit_v1/conclusions.md
8baa9d66f58436037a1de79a8b7407a8a0662a70ddf1e0d2d9ea3c6a25b0d492  european_easy_hurdle_v1/results.md
```

Document paths in the second block are relative to the parent publication
readiness directory. This static audit is not another model training or
world-dynamics result. Independent roles, deployment, Stage5C and SMC remain
unchanged; no metric, effective-second, true-3D or foundation claim.
