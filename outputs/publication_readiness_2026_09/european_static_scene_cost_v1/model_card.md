# Static Scene-Context Cost Head

Research-only action-cost selector,not a new trajectory world-dynamics model.
54newheads/108,000updates,26,309parameters each. Existing25,157weights and
draws are matched;an additional1,152zero-initialized projection weights admit
18static/past/proposed-rollout image-region fields. Shared-reference repair is
not stacked onto this ablation. Original control remains frozen and replayable.

Inputs:380causal motion/rollout fields,18scene-region fields,7action identities.
No agent/scene ID embedding,future endpoint,label-validity mask,central
velocity,test statistic or future-built goal. Candidate rollouts are forecasts,
not targets. Only physical fitting candidates set common normalizers.
Future data are used only for supervised fitting costs and fixed evaluation;
unknown outcomes remain eligible at inference and absent from regression loss.

Outputs:five bounded nonnegative benefit/harm/reference/easy-reference/easy-harm
moments. Same positive-gain and predicted2%ratio rule,samefixed1/2/4sensitivity.
Expected-cost inference is not independently calibrated risk control.

Known-label mixed gain0.982769%;scene-minus-parent+0.428982pp with positive
conditional interval. Yet mixed-minus-physical is-0.093829pp with negative
interval,completion bounds are not positive,and required-risk support23/186.
Do not deploy. Keep existing floor. These SOURCE inner-held data were already
exposed,with one frozen forecast seed and three controller seeds.

CPU4/interop1/workers0,checkpoint100/heartbeat50,10GiBreserve,12hhealthy cap.
All54endpoints freeze before144view readout;exact replay and metric bundle
verification are required. Weights/source caches remain ignored/private.
Study-area polygons are not verified walkable maps or metric calibration.
No seconds,true3D,foundation,independent confirmation,Stage5C orSMC claim.
