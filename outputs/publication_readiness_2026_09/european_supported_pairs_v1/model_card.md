# Supported-Pair Risk Head

Status: research-only risk-controller ablation. Not a new trajectory dynamics
model, not deployable and not submission ready.

| Item | Description |
|---|---|
| Model | Two linear layers with GELU, 355 causal features, width 64, three outputs. |
| Size | 22,979 parameters per head; 36 heads over three folds/seeds and two candidates/events. |
| Outputs | Conditional reference-error moment, harm probability and conditional severity; bounded harm moment. |
| Inputs | Same past-only features and causal baseline/candidate rollout envelope as the frozen control. |
| Supervision | Fitting-only cross-fitted errors. Future labels are not inference inputs. |
| Changed factor | Remove undefined event-mass rows before within-locality cyclic ranking pairs. |
| Unchanged objectives | Moment MSE, occurrence BCE, positive severity MSE and margin-weighted rank logistic. |
| Budget | 2,000 updates/head, fixed coefficient 1, AdamW, identical sampler sequence. |
| Runtime | Native arm64 CPU4/inter-op1/workers0, atomic checkpoints and resume. |
| Data role | European Squares opened development, four fitting/eight complete-chain-excluded localities. |
| Units | Released detector-track image pixels; 8 observed/12 predicted rawstride12 positions. |
| Preservation | All neural positive-easy checks pass, but zero-reference protection still fails in 11 views. |
| Strong comparison | No positive neural all-ADE point against equally protected damping in 18 comparisons. |

Every registered comparison is retained. Three seeds and conditional locality
bootstrap are not independent confirmation. Shared upstream predictors and source
exposure limit generalization claims. The source-role identities and checkpoint
hashes are recorded in verification and training receipts; binaries are not in Git.

No historical Stage37 recertification, new JEPA/Transformer fit, physical-safety
guarantee, metric/seconds claim, true-3D or foundation claim follows. No deployment
changes, Stage5C execution or SMC. The synthetic counterexample is a separate
target-definition diagnostic, not measured model success on real trajectories.
