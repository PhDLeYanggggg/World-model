# User Action Required

- reason: stage14_follow_up

## Actions

- EWAP t+100 per-agent rows are now evaluable in Stage14, but deterministic improvement is only about 0.008 and does not pass the 5% long-horizon gate.
- Provide/convert additional pedestrian or drone long-horizon data such as SDD/OpenTraj if available.
- Reboot the Mac if the old OpenMP/SHM-stuck PIDs remain visible after SIGKILL; the fixed runner now avoids torch resource probing and uses inline Stage14 execution.
