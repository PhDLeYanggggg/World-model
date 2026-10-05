# CREATE execution amendment

The registered temporal auxiliary experiment moves to isolated M3W CPU compute
because local storage is below its checkpoint reserve. This is an execution-location
amendment, not a new hypothesis, a split change, or an evaluation-rule change.

The original optimizer, three arms, 72 source heads, three seeds, 2,000 updates,
loss weight, query sampling, 2% risk budget and separate seven-arm readout remain
unchanged. The portable runner invokes the original `train` and `api.fit` functions.
No new forecaster is trained. The original training registration remains binding.

Only original source TRAIN arrays and their TRAIN-fitted preprocessing are streamed
losslessly in memory. Seeds share one identical packet per source group. Features,
moment labels and temporal labels stay distinct; no future target becomes an input.
Packets are checksum-verified and pickle loading is disabled. Validation rows and
independent roles are not transported. This does not erase prior source exposure.

The existing M3W-only CPU runtime must have a successful real synthetic optimizer
and exact-resume receipt, corroborated by completed Slurm accounting. Science runs
only on compute nodes: one task, four CPU threads, zero DataLoader workers, 16 GiB
RAM. Pilot wall time is two hours; full-run wall time is at most twelve hours.
Other project environments and jobs remain untouched.

The 10 GiB reserve, 256 MiB checkpoint cap and 2 MiB atomic headroom remain in force.
CREATE uses personal Ceph quota, not shared-filesystem free space. Input transport
has an additional 4 GiB total cap and 512 MiB packet cap; it stops rather than
weakening the reserve. No large local temporary data or model files are created.

Before full training, the real-data pilot must complete 100 updates in each arm,
match sampler draws, reproduce interrupted/resumed versus uninterrupted training
exactly on that runtime, fit the memory budget and project a feasible full run.
The estimate is not an observed full runtime or a scientific result. Existing
checkpoint identities are verified before resume. Submission intents prevent
duplicate jobs after ambiguous network outcomes.

No validation selection, independent evaluation, deployment change, Stage5C or SMC
is authorized by this amendment. Runtime success alone cannot establish model lift.
