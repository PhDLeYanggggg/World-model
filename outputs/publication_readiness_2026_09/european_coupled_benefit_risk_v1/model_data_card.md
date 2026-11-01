# Coupled neural cost-head model/data card

Task:predict benefit,harm and easy-harm moments for a frozen seven-action bank;select optional baseline/neural interventions.
This is a neural decision component,not new trajectory dynamics,JEPA pretraining,latent rollout or a foundation world model.
Inputs:existing462-field past trajectory/neighbor/detector/scene/rollout context,old predicted moments,causal masks and query grouping.
Future endpoints,central velocity,test-derived goals and future support are not input fields. Loss targets and fitting-only RMS use supervised training labels.
References R/easy_R remain fixed. Released benefit/harm lie in the declared envelope;this alone is not complete physical or probabilistic moment coherence.
Architecture:width64 selected-set critic and scalar benefit branch;30,857 nominal parameters per arm. Protected branch is unused.
Training:36 fits,3 critic seeds,6 reused source contexts,2,000 updates each;two complete exact refits.
Data:149,510 unique detector-silver target rows,74 recordings,12 exposed European SOURCE localities;whole-pipeline OOF=false.
Observation protocol:8 past/12 future steps,raw stride12,oblique local-image coordinates. No verified metric/seconds/human-gold/true3D claim.
Three critic seeds are not three independently trained forecast producers. There is no independent calibration/confirmation result.
Result:registered primary and original recording-risk requirements fail. Deployment unchanged;not CCF-A ready.
Private data,features,weights and per-row choices are excluded from Git. Public aggregate bundles and source/checkpoint hashes support local reproducibility.
Local native arm64 CPU4/interop1/workers0. No new CREATE job,worker multiprocessing or resource probing.
Stage5C execution=false;SMC=false.
