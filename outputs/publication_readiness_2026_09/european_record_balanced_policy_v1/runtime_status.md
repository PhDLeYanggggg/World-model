# Runtime and CREATE Observation

Fresh observations on29October2026, before fitting readout.

The native arm64 CPU pilot completed60realupdates in14.278188s,peakRSS
2,679,816,192B,PID27689,exit0. Balanced10+10resume exactly matches20direct
updates; neutral20updates reproduce the frozen parent's20update training state.
The pilot predicts roughly956s for all18fits; this is an estimate,not completion.
Full training ran under PID28251 with CPU4, interop1, workers0, atomic checkpoints,
heartbeat and a process lock. All 18 fits and 36,000 updates completed with
captured exit0 in 732.451502s, peak RSS 2,821,308,416B. Final weights are frozen
before readout. Model and split scope were not reduced for speed.

A fresh read-only CREATE query ran10:02:04to10:02:07UTC. It used the existing
authorized SSH route and account-filtered `squeue --clusters=create` command.
Explicit unique start and end markers verify that the remote command executed
and returned0. The query listed zero current jobs for the selected account.
This establishes a queue snapshot,not the completion of historical jobs.
No job was submitted,restarted,cancelled or otherwise modified.

Private receipt SHA256:
`f9b4619438c07cce89b61b817bd95dcc8764ba8446d361f118ad9bb2b90c5cec`.
The new observer rejects a login banner plus exit0 without execution markers;
three tests cover authentication-only output,empty verified queue and invalid
markers. The older uncertain receipt remains historical,not rewritten.

The current experiment needs about 3GB RAM and runs its numerical training
locally. Its authoritative loader streams frozen source-TRAIN NPZ packets from
CREATE into local memory over the existing read-only SSH route. Packet byte
counts, SHA256 hashes, owner and TRAIN manifest are checked before use. This is
not a wholly local data store and no new packet disk cache is created. Only the
registered fitting recordings enter this experiment; historically exposed
inner-held recordings in the source-TRAIN packets are not evaluated here.
Independent selection, calibration and confirmation roles remain unopened.

The real pilot includes packet access and supports local execution at this
scale; no new HPC training job is needed. Free disk was 22GiB before admission;
the 10GiB reserve and 128MiB checkpoint cap remain active. These runtime checks
do not establish predictive quality or submission readiness.
