# Ranking-Augmented Risk Head

Status:research-only controller ablation;not deployable or submission ready.

| Item | Description |
|---|---|
| Model | Two-layer GELU MLP,355 causal features,width64,three factor outputs. |
| Size |22,979 parameters per head;36 fitted heads over folds,seeds,candidates,events. |
| Outputs | Reference-error moment,harm-occurrence probability,conditional severity;bounded harm moment. |
| Inference | Same past-only feature vector and candidate/baseline rollout envelope as the frozen control. |
| Labels | Fitting-only cross-fitted errors,not inference features. Unknown labels excluded from sampling. |
| Added objective | Within-locality margin-weighted pairwise logistic risk ordering,coefficient1. |
| Unchanged losses | Moment MSE,occurrence BCE,positive-only severity MSE. |
| Budget |2,000 updates/head,AdamW,three seeds,identical control minibatch draws. |
| Training role | European Squares opened source development;four fitting/eight excluded localities per fit. |
| Units | Detector-track image pixels;8 observed/12 predicted positions at rawstride12. |
| Expected use | Controlled method diagnosis and source-development research only. |
| Prohibited interpretation | Independent safety proof,new trajectory dynamics success,true3D,foundation or metric/seconds prediction. |

All36 checkpoints replay on the first4,096 excluded-index rows each;full metrics
also reproduce. Checkpoints remain local and hash-indexed in training_metrics.json.
The absolute source-role and model-chain identities are bound in verification.json.
Publishing these records does not make private data/checkpoints available from Git.

The new neural controller fails to beat matched protected damping in every
all-ADE point comparison,and still harms zero-CV rows. Modified damping heads
also fail easy preservation in three views. No current deployment is replaced.
The empty zero-error support in six readout views is not a passed-event guarantee.

There is no new latent rollout,JEPA pretraining,Transformer fitting,SMC or Stage5C
execution. Multimodal/interaction/physical-safety contribution is not established
by this risk-head-only experiment. Independent calibration and confirmation remain
closed,and the complete research objective remains unfinished.
