# Causal-anchor control model/data card

Model:closed-form ordinary least-squares velocity over last4/8 of8 observed2D positions.
Compared state estimate:raw last observation versus causal fitted line evaluated at current observation.
Output:12 deterministic future local-image2D positions. No learned weights,no neural training,no target used for inference.
Candidate inputs:history only. Original past-only eligibility/floor and producer43 identity remain fixed.
No new scene RGB,goal endpoints,central velocity or future support is supplied as input.

Data:149,510 unique detector-silver target rows,74 unique recordings,12 exposed SOURCE localities,24 reused fitting views.
Obs8/pred12/rawstride12. Local image coordinates,oblique EuropeanSquares;not the SDD raw-frame t50 benchmark.
No verified homography,scale,seconds,physical safety,true3D or foundation status. Missing labels retain valid masks.
Future labels are error targets only. Unknown-completion envelopes are not imputed trajectories.

Result:registered fitted-position repair negative,4/7 fitting checks pass,0/148 supported original record/context risk views for all4 arms.
Independent evidence:new none. External/SDD deployment safety:new not_run. Decision:no deployment change.
Arithmetic diagnosis:post-hoc,error identity only,not a new primary metric or motion/noise identification.
Runtime:native arm64 CPU4/interop1/workers0,no resource probing,multiprocessing or remote scheduler changes.
Reproducibility:registered source/config/test hashes,immutable24group receipts,aggregate ZIP and complete separate prediction/replay.
Public distribution:code,protocols,aggregated metrics and report only;private trajectories/caches/media/weights excluded.
Stage5C execution=false;SMC=false;submission-ready=false.
