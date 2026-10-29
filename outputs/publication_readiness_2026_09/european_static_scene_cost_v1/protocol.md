# Static Scene-Context Cost Ablation

Prospective29October2026. The completed TRAIN audit finds exact camera mapping
for14image-region polygons,74fitting recordings,149,510unique rows/148context-
record views. No current point is outside the expanded[-.5,1.5]polygon bounding
box. This only rules out gross scale mismatch,not verifies calibration or scene
semantics. Missingness is not monotonically related to polygon distance.

Test whether18lawful static/past/predicted location features add predictive
value absent from the original globally translation-invariant motion features.
Keep original raw pixel midpoints,8observed/12predicted/rawstride12,all rows,
forecast bank,loss,optimizer,2,000updates,draws and2%risk rule unchanged.
Do not clamp forecasts or claim the place polygon is walkable/obstacle/viewport.

Append18fields to380motion fields. A zero-initialized18x64projection enters
the existing hidden layer. Existing25,157weights and random stream stay exact;
the1,152additional weights are disclosed,26,309parameters total. Disabled
projection is a parameter-matched control:full2,000updates must reproduce
original network/Adam moments/draws/traces exactly. The original frozen54fits
are cached-verified controls,not54newly trained models. This experiment does
not combine the negative shared-reference repair with the scene change.

Scene fields:normalized current/past-start position,current/mean velocity,
past-start/current signed polygon distance,distance change,history inside
fraction,rollout distance atsteps1/6/12,rollout inside fraction,min/mean rollout
distance. Only static per-camera geometry,past observations and the proposed
rollout enter. Each camera binds by exact raw member ID,not locality ID.
Physical fitting candidates alone set scene normalizers. Unknown fitting
features remain,but unknown labels never enter regression. All unknown readout
rows remain eligible. No future support,endpoint,goal,test statistic or reserved
outcome enters inference or eligibility. Scene identity itself is not a feature.

Sixcontexts x threeseeds x threebanks =54freshheads/108,000primary updates.
Require real all-bank10+10/direct20pilot and full disabled neutral. Match
original network initialization,row draws and RNG,then freeze all54endpoints
before readout. Report every fixed1/2/4harm sensitivity,all required recording
risk views,mixed-minus-physical and scene-minus-no-scene paired intervals.
No best-seed,held threshold or favorable-sensitivity promotion. Same-count
controls are utility-only;empty risk is not_estimable. Scalar cost calibration,
scene features and statistical significance do not themselves imply safety.

CPU4/interop1/workers0;checkpoint100/heartbeat50,10GiBreserve,12hhealthy cap.
Independent selection/calibration/confirmation/DroneCrowd stay closed. Source
inner-held is exposed exploratory development,not whole-pipeline OOF/fresh test.
No new trajectory producer,scene-joint optimization,metric/seconds,true3D,
foundation,Stage5C orSMC claim/execution. Deployment remains unchanged unless
the original full-policy all-required-recording contract actually passes.
