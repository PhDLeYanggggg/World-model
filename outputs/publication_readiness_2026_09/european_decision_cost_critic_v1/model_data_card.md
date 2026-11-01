# Decision-Cost Critic: Model And Data Card

Architecture:the frozen selected-query coupled critic,30,857 parameters,width64.
462 past-only fields per candidate,seven fixed forecasts and original selected-set
query pooling. Predicts benefit,harm and easy harm;the original all/easy references
remain fixed. No new trajectory encoder,JEPA,Transformer,latent rollout or correction.

Training:18 fresh critics at2,000steps each,seeds17/29/43,AdamW,original draws and
initialization,plus one2,000-step exact full refit. Cached18 regression controls
reproduce on identical input hashes. Loss adds SOURCE expected decision costs and
whole-record dual updates to the original calibration auxiliary. All1,332 final
expected training constraints remain positive. Training completion is not a risk pass.

Inference accepts only the frozen causal schema. No future endpoint,target goal,
central velocity,label-support mask,training dual or test normalization is an input.
It retains original deterministic eligibility,2%predicted all/easy constraints and
current-query joint optimization. Solver rejection keeps the feasible parent.
Actual original recording-risk evaluation,not the soft training objective,controls advancement.

Data:149,510unique targets,74 source-qualified recordings,12exposed SOURCE localities.
72context/critic-seed fitting views repeat observations. EuropeanSquares oblique
detector-silver image coordinates,obs8/pred12/rawstride12. This is not SDDt50,
human gold,verified meters/seconds,true3D or a foundation-model result.
Wholly unknown future rows are retained using causal disagreement bounds. Missing
endpoints remain missing;wholly unknown bounds do not resolve interior missingness.

All present results are fitting resubstitution on previously exposed sources.
Not independent confirmation,whole-pipeline OOF or three forecasting-producer seeds.
No new independent data role or reserved DroneCrowd label is opened. No raw data,
features,latents,images or checkpoints belong in the public evidence bundle.
Deployment remains unchanged. Stage5C and SMC remain disabled. Submission readiness:false.
