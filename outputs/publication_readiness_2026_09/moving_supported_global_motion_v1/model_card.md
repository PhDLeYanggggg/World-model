# Supported Global-Past-Motion Cost Heads

Status: prospectively registered development experiment. Completion and outcome
are determined by training_freeze,verification,inference_freeze and analysis,
not this descriptive card. No deployment change is authorized by fitting loss.

Architecture:19,430parameter Torch cost/support head. Local causal120 feeds
64/32conditional hidden layers. Fifty action-invariant history/neighbor features
feed separate32wide reference and observation branches. Three zero-initialized
global48projections enter the respective hidden layers. Local ablations mask
global input but retain identical parameter count;masked branches are inactive.
It is not new JEPA/Transformer
pretraining or a newly trained trajectory forecaster.

Outputs: conditional nonnegative benefit,harm,floor reference,easy reference and
easy harm,plus predicted future-label observation probability. Coherent envelope,
benefit/reference and easy-moment constraints match the existing role-separated
model. Unknown-label moments use causal path-disagreement bounds. These are
expected prediction-error contributions,not physical safety guarantees.

Inputs contain only past trajectories,neighbor/crowd motion and frozen candidate
rollout diagnostics. Candidate future predictions are causal model outputs,
not ground-truth future. No future target/validity,endpoint goal,test statistics,
central-difference velocity or source identity is an inference feature. The
past coverage summary is allowed; future annotation support is a training label.

Training: three risk seeds,96fixed final models at4096updates,matched source-held
fitting. Own-only and fixed50%producer-A donor augmentation crossed with masked
and admitted global context. Same loss,batch512,AdamW .001,underharm4,gradient
clip5. Donor support changes teacher/candidate population too. No held epoch,
threshold or amplitude selection. Registered all96full exact refits check same
implementation reproducibility,not independent statistical replication.

Decision policies preserve original independent/frame-uniform/joint controls
and2%all/easy positive-error budgets. Empty support is NA. A fitted expected-cost
constraint is not a verified realized-risk bound. Actual original-contract
checks are reported for all72repeated views. No identity certification.

Intended use: mechanism/ablation evidence on already-exposed MovingDroneCrowd++
own-B data. Not validated production use,not independent confirmation,not metric,
seconds,true3D,foundation or human-gold evidence. No Stage5C execution or SMC.
