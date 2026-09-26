# Resource and Execution Observation

## Fresh Preflight
Local and GitHub parent ebf198f3 matched before this experiment. CREATE was
queried read-only successfully; private receipt SHA256
9d08d0786e1c141bb4568bd570c40cff791bed3f3fd32bc20064ed34683d2d81.
No remote jobs were submitted, cancelled or modified. Native arm64 Python
3.11.1 and19GiB free disk observed;10GiB reserve enforced by the runner.
Support PID24112 completed all144 views normally. Observed source-loading
RSS was about11.3GiB, within the48GiB machine. This is an observation, not
a maximum-memory guarantee for every future phase.

## Actual Training Pilot
Registration1b8f5800 and support fca1dcbf were pushed before training.
PID24814 completed100 cost-only updates. Fitting0.11604100000113249 seconds;
phase wall15.771427833009511 seconds including loading, excluding ancestry
preflight. Cost loss on the fixed fitting batch0.5136901140 to0.3332728148.
Parameters12,899;unknown-label rows sampled0. CPU4/interop1/workers0.
This is real native-Torch training, not only an import or NumPy fallback.

Multiplying the cost-only pilot rate by864000 updates gives1002.59 seconds
of approximate fitting work, excluding source loading/inference/verification
and before measuring auxiliary-arm runtime. It is not an end-to-end ETA.
Local execution is reasonable; checkpoint/resume and heartbeats are active.
The100-update pilot is continued to2000, not counted as a separate full model.

Full training and held readout are not complete at this observation. Results
and final measured runtime will be recorded after authoritative termination.
