# Method Context and Contribution Boundary

Primary-source check: 2026-09-27, before this experiment's comparative readout.

AgentFormer explicitly addresses the loss of agent identity when multi-agent
trajectories are flattened. It distinguishes within-agent and across-agent
attention, and also warns that separately summarizing temporal and social
dimensions can lose information. Our temporal-then-interaction repair is a
small controlled test, not AgentFormer, not a new discovery of identity-aware
attention, and not proof that factorization is optimal.
[Yuan et al., ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_AgentFormer_Agent-Aware_Transformers_for_Socio-Temporal_Multi-Agent_Forecasting_ICCV_2021_paper.html).

Scene Transformer already models temporal and multi-agent context with a
scene-centric attention architecture and permutation-equivariant agent
representation. Its joint-forecasting formulation is distinct from our current
per-target deterministic bounded forecaster. A pooling/topology gain here
would not demonstrate joint future consistency.
[Ngiam et al., Scene Transformer](https://arxiv.org/abs/2106.08417).

The proposed M3W contribution must instead establish when a neural forecast
should replace a strong physical-motion baseline, using relative gain/harm
learning, scene-level joint intervention, independent locality risk calibration
and support-aware fallback. This experiment only tests a predictor prerequisite.
It neither establishes the novelty nor the empirical success of that policy.

These are method-context comparisons, not new same-data reproductions of the
two papers. No stochastic/generative branch is enabled. Stage5C/SMC remain off.
