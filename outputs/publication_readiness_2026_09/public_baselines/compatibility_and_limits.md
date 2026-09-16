# Public Forecasting Controls: Source Audit and Causal Adapter

Date: 2026-09-16. Status: implementation verified; real forecasting comparison **not_run**. This is not an EqMotion paper reproduction, a new predictive gain, or a completed strong-baseline study.

Update after the user's 8/12 decision: the local Transformer now has a completed,
negative three-seed [real development experiment](../8to12_development_v1/results.md),
and a separate robust-loss ablation is also complete without safe positive gain. The unapproved-protocol references
below describe this source audit's earlier snapshot. They no longer block the
versioned development study; independent confirmation is still unapproved.
EqMotion itself has not yet been fitted on these real labels. Its K=1 adaptation
and incomplete-neighbor exclusions still require explicit matched controls.

## Source Identity

The [official EqMotion repository](https://github.com/MediaBrain-SJTU/EqMotion) is pinned to `5aec2e0b61c511fa93a24138dd90da59a089084b`. Eight code/documentation files, 67,758 bytes, were fetched without datasets or pretrained weights. Their author Git-blob identities and local SHA256 values are recorded in `eqmotion_source_identity.json`; the MIT license is retained beside the ignored source. The adapter verifies every pinned file before loading code. Only the two model modules execute, under a private namespace; the import binding is isolated without changing their model arithmetic. Third-party code is not committed into this project.

[EqMotion](https://arxiv.org/abs/2303.10876) is a published equivariant motion-prediction method, not evidence that M3W's relative geometry is novel. This implementation is one candidate public control, not a claim to cover current state of the art.

## Protocol Differences That Matter

| Finding | Required treatment here |
| --- | --- |
| The pinned pedestrian model has 20 parallel prediction heads. | Do not compare a label-selected minimum over 20 with a single emitted forecast. |
| The pinned training entry saves its best checkpoint based on test FDE. | Do not execute that selection loop. Use fit-only training and a separately declared development role. |
| The author's preprocessing determines valid agents using past and future availability. | Do not import it. Build context from observable past only; missing future is a scoring mask, not inference membership. |
| Original entry assumes CUDA, multiple loader workers and processed-array paths. | Use the project's explicit CPU/MPS, workers=0, atomic checkpoint and provenance path. |
| AgentFormer publishes a normalization correction and revised results. | Do not quote its original numbers as the corrected comparator or pool them with our local metrics. |

Sources: [pinned EqMotion model](https://github.com/MediaBrain-SJTU/EqMotion/blob/5aec2e0b61c511fa93a24138dd90da59a089084b/eth_ucy/model_t.py), [training/evaluation entry](https://github.com/MediaBrain-SJTU/EqMotion/blob/5aec2e0b61c511fa93a24138dd90da59a089084b/main_eth_diverse.py), [preprocessor](https://github.com/MediaBrain-SJTU/EqMotion/blob/5aec2e0b61c511fa93a24138dd90da59a089084b/eth_ucy/preprocessor.py), [AgentFormer author notice](https://github.com/Khrylx/AgentFormer#important-note). These are observations about the checked release, not a re-analysis of either paper's experiments.

## Implemented Control

`EqMotionFixedHead` emits one head fixed before fitting, default head 0. Its loss is the same masked, past-normalized coordinate MSE used by the local forecasting backend. Other output-head parameters are frozen, although the unchanged core still computes the 20-head bank. This changes the training objective from the author's diverse-prediction entry. It must be called **EqMotion-core fixed-head adaptation (K=1)**, not the paper's deterministic result, not minADE20/minFDE20, and not an official reproduction. No head is chosen by ground-truth future error at inference. There is no stochastic latent rollout, Stage5C execution or SMC.

Inputs contain the target agent first, followed by fully observed, timestamp-aligned neighbors, then padding. Displacements are backward differences within the past window; the first displacement copies the second, as in the author input calculation. There is no central velocity or future endpoint feature. The official graph only supports agent-level masks; partially observed or misaligned neighbors are excluded according to past support, not silently filled with zeros as observations. Target agents are not dropped for missing future labels.

The fixed model refuses partial ego history, padded/ragged prediction requests, nonuniform observation/request spacing, post-current timestamps, unexpected input fields and nonfinite observed values. It therefore needs a separate fixed-shape model for a different horizon/grid. The 8/12 configuration is an engineering proposal, not approval of the main scientific horizon or a verified physical-time claim.

The shared trainer now constructs either the local Transformer or this adapter. Checkpoints bind the official-file hashes, source manifest, adapter code, training backend, architecture, head identity, fit recordings and experiment protocol. Verified loaders reject missing or changed dependencies, including an output-head change inconsistent with the frozen training identity. The same verified predictor can produce held-out-fold realized benefit/harm targets and enter the five-arm development evaluator. Existing checkpoints from an older backend hash are not silently relabelled as current-version experiments.

## Fresh Engineering Evidence

- 17 adapter tests; combined targeted regression: **136 passed in 24.30 s**, zero skips in this local run. A machine without the optional source skips the source-dependent tests explicitly; that is not a verified integration.
- Direct author-module invocation and the adapted fixed head give exactly equal outputs with the same weights/inputs. Tests cover backward gradients, rigid-transform/neighbor-order behavior, future mutation, neighbor masks, source tampering, checkpoint provenance, OOF cost construction and resume.
- CPU synthetic fitting: continuous 8 updates equals interrupted 3+5 updates, including parameters, losses, sampling and optimizer values. A separate-process check exercises the actual training entry, not just an in-memory loop.
- MPS separate-process synthetic recovery: **1 passed in 11.66 s**, exact parameter/loss/optimizer equality. The sandbox initially denied Metal initialization despite macOS 15.3.1; explicit execution outside the sandbox worked. A subsequent strict state check found the old restore loop moved AdamW's non-capturable step counter to MPS. The loop was removed so PyTorch applies its own state-placement policy; the strict check then passed. No implicit CPU fallback was enabled.
- CPU and MPS real-input checks each cover **27 scene requests / 330 target agents** with random weights, no real labels and no accuracy computation. All checked forecasts/features are invariant to replacement of every post-current position. Of 2,445 observed neighbor slots, 2,128 have a complete matching past grid; 317 are not admitted to this core. Counts describe repeated ego-centered input slots, not unique people.
- The existing local-Transformer input checks were refreshed after the backend edit: 24 queries / 345 agents and 23 queries / 274 agents for the five-arm evaluator, with zero future-label calls. Earlier training reports remain evidence for their recorded code version.

See `verification.json`, `eqmotion_real_input_checks_cpu.json`, and `eqmotion_real_input_checks_mps.json`. These results use small synthetic fitting or random-weight real inputs; they do **not** show trained EqMotion accuracy, long-run stability, superiority, or independent generalization. The historical full-suite result remains 1,870 pass / 1 unrelated data-lake fixture failure; it was not rerun or turned into all-green.

## Remaining Comparative Evidence

| Control | Current evidence | Missing before a paper claim |
| --- | --- | --- |
| Strong causal floor | Reader implements seven causal rollouts. | Fit/development selection under approved roles and a new accuracy table. |
| Local past-context Transformer | Synthetic training/recovery and causal-input checks. | Clean real fit, OOF costs and development comparison. |
| EqMotion-core K=1 adaptation | Pinned source; actual CPU/MPS fitting and recovery on synthetic fixtures. | Matched real training, tuned budget on development only, three seeds and scene-block uncertainty. |
| Published diverse EqMotion protocol | Code reviewed, no original training/evaluation executed. | Explicit matched sampling budget, loss and metric protocol; clean retraining rather than exposed checkpoints. |
| AgentFormer | Author documentation and correction notice reviewed. | Version-pinned adapter and an explicitly scoped stochastic comparator protocol; **not_run**, no model/data/weights fetched. |

For fair predictor comparison, report the common fully observed-neighbor support and an additional full-observed-context local-model control. Otherwise any difference may reflect missing-history handling rather than model quality. For intervention comparisons, use the exact same frozen predictor outputs across all five policies and report realized coverage as well as budget caps. Fixed-head adaptation alone does not close the published-baseline requirement.

The real protocol remains unapproved: main horizon, record/physical-scene roles, aggregation/error unit and risk budgets still require explicit decisions. Previously exposed ETH/UCY recordings remain development material, not untouched confirmation. CREATE access remains unresolved. The next permissible real experiment follows protocol approval; no replacement deployment or metric/seconds/3D/foundation claim is made here.
