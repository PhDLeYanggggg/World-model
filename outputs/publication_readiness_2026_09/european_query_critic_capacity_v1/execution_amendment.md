# Cold-replay execution amendment

The registered scientific diagnostic and its completed results do not change.
After the parent policy readout completed, the old entry point was actually
rerun and refused with exit1: it recomputed
`parent_policy_readout_complete_at_registration` as true, although the event
was false when registration was created. The old immutable registration,
source, two completed diagnostic runs and result hashes are preserved.

This separate execution adapter changes only that creation-time event check.
It restores the recorded boolean when comparing a proposed registration to the
original, then requires exact equality of every other field. Original source,
input, role, budget, group and result hashes remain checked. Missing/nonboolean
event metadata and any scientific or role change still refuse. This is not a
permission bypass or a retrospective scientific amendment.

The adapter, helper, tests and this note are hash-registered and committed
before running a new complete diagnostic replay. The replay executes the
unchanged 72-group algorithm, requires all groups and summary to match, and
adds a separate PID/runtime/hash receipt. It never overwrites original results
or claims new training, a new policy, independent evidence or risk calibration.

```sh
.venv-pytorch/bin/python scripts/replay_m3w_query_critic_capacity.py verify
```
