# Policy-Region Cost Moment Head

Research-only expected-cost controller,not a newly trained trajectory dynamics
model. The54new heads each have26,309parameters and match their old scene-head
initial weights,preprocessing,row draws,optimizer and2,000-step schedule.
All108,000primary updates are actual native arm64PyTorch training. The frozen
trajectory producer is unchanged. Three seeds are controller seeds,not three
independently retrained forecasting pipelines.

Inputs remain380causal motion/rollout fields,18static/past/rollout region fields
and7action identities. No future endpoint,central velocity,future-validity mask,
held outcome,scene/agent-ID embedding,test statistic or endpoint-built goal
enters inference. Supervised future costs are fitting labels only. Candidate
rollouts are forecasts,not target trajectories. Unknown teacher-selected rows
have no regression/calibration label and remain unknown at evaluation.

Outputs are structured nonnegative benefit,harm,reference,easy-reference and
easy-harm cost estimates. The unchanged inference gate requires positive
predicted gain and predicted all/easy harm ratios<=.02. Fixed1/2/4sensitivity
coefficients are controls,not a menu selected after held readout.

The only primary change is fixed-weight1aggregate H/R/ER/EH moment training
inside the old teacher's selected fitting-recording regions. Recording IDs
group the training loss only. Fitting centroids are not independent calibration;
minibatch regional means introduce variance,and new decisions may leave the
old regions. This does not implement or establish a calibration theorem.

First readout:known mixed ADE gain0.943808%,mixed-minus-physical+0.038578pp
with conditional95%CI[-0.087280,+0.166017]. Completion-lower neural increment
is-0.656490pp,CI[-1.756899,-0.009889]. Only19/186required record/head views
support the full risk criterion. Primary gates fail;do not deploy this head.
See the separately verified evidence/readout for exact replay and mechanism
diagnosis. No threshold or checkpoint is chosen from these outcomes.

Obs8/pred12/rawstride12,EuropeanSquares SOURCE oblique webcam detector-track
pixel box midpoints. Previously exposed SOURCE inner-held data are development,
not independent confirmation or whole-pipeline OOF. Repeated record/head views
are not independent recordings. The static polygons are region proxies,not
verified walkability,homography or metric scale. Current M3W remains a2.5D
trajectory scaffold,not true3D or a foundation model. Raw-frame units are not
effective seconds. No Stage5C execution orSMC is enabled.

CPU4/interop1/workers0,checkpoint100/heartbeat50,10GiBdisk reserve and12hhealthy
resume boundary. Checkpoints and licensed inputs remain private/ignored;
public aggregate replay is not raw-data end-to-end reproduction.
