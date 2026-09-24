# CREATE Read-Only Refresh, 2026-09-24

The existing authorized short SSH connection was attempted for the same read-only
queue query used in the prior handoff. It returned exit255 with authentication
denied. The remaining serial partition/reservation queries were not run after
that failure. This does not establish that any remote job has stopped.

- `fresh_run`: one bounded read-only connection attempt, unsuccessful.
- `not_run`: current queue inspection, remote M3W asset inventory, data transfer,
  runtime setup, job submission, cancellation or changes to credentials.
- The previous day's successful connection and maintenance snapshot remain dated
  historical evidence, not a current scheduler or authentication status.
- The simulation project was not accessed or modified. No key contents,
  passwords, OTPs or tokens were read or exported. No SSH configuration changed.
- Local assets suffice for the missing twelve small forest cost heads; this
  connection limitation does not block the current experiment. No HPC job is
  needed for that experiment, and no protected workload is displaced.

Local-only generated receipt:
`data/stage_cvpr2027_experiments/create_handoff_20260924/read_only_refresh.json`

SHA256: `ae013031420de8a7ac60775ac6c846b5632be08fd3b86fe5c158dfd288407aeb`.
The receipt is ignored by Git. This public record deliberately omits account,
key path and internal system identifiers. Authentication failure is not attributed
to an unverified cause, and is not a scientific/model result.
