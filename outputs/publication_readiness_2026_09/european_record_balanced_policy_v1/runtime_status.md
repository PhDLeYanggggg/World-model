# Runtime and CREATE Observation

Fresh observations on29October2026, before fitting readout.

The native arm64 CPU pilot completed60realupdates in14.278188s,peakRSS
2,679,816,192B,PID27689,exit0. Balanced10+10resume exactly matches20direct
updates; neutral20updates reproduce the frozen parent's20update training state.
The pilot predicts roughly956s for all18fits; this is an estimate,not completion.
Full training starts underPID28251 withCPU4,interop1,workers0,atomic checkpoints,
heartbeat and a process lock. Model and split scope are not reduced for speed.

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

The current experiment needs about3GB RAM and locally available data,so local
execution is appropriate. Free disk was22GiB before admission; the10GiBreserve
and128MiBcheckpoint cap remain active. This observation neither allocates HPC
resources nor establishes predictive quality or submission readiness.
