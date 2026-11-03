# Resource Interruption, Not A Completed Duration Result

## Status

Fresh real Torch training ran under PID60218 and exited1 when the existing
10GiB free-disk guard fired. This was not a timeout, OpenMP failure or a live
stalled process. The process is no longer present and its tool session has
returned the terminal exit. No automatic duplicate or lower-budget run began.

Six of eighteen full endpoints are complete. A seventh checkpoint contains
14,600/16,384 updates. Seven original-budget prefixes are exact; the first
8,000-update full from-scratch refit is also exact. Checkpoints preserve
81,608 primary optimizer updates, including replayed original prefixes, plus
8,000 refit updates. Any iterations after the last atomic checkpoint are not
counted here. Do not report232,320updates or18complete models.

No aggregate duration verdict, full fitting replay or new H policy evaluation
has run. The six partial-job readouts are not selected or averaged to replace
the full roster. The original feature primary remains inconclusive and its
complete risk remains failed. No deployment or scientific gate is promoted.

## Evidence And Recovery

- Failed log SHA256: `ac82e4c4dbbe4a086fcfe68d708de72f1cd40c7dabf4383bcb599fde0149d43d`.
- Private recovery observation SHA256: `13f3e248ecd7948469d1b848d0eabf4258f012428385d697e82a0c98c2b989f1`.
- Private backup receipt SHA256: `6d6b36a85ff34882cc81cc66276dce718f5bf540202c0657641af7e5e9d62f3c`.

The storage snapshot has less than4GB free locally; this probe's private
directory itself occupies about5.3MiB. The source of the wider volume pressure
has not been established. No unrelated old data, cache or user file was deleted,
and no scientific budget or disk reserve was reduced to force a completion.

Fresh CREATE SSH, queue, CPU-partition and filesystem queries all return0.
The existing owned M3W root is verified. Six named small receipts were inspected;
they are not current scientific compatibility checks. No M3W-named queued job
appears in this snapshot. Ordinary CPU has a48-hour maximum. Home quota is
50,000,000,000bytes with43,457,188,010used: remaining6,542,811,990bytes,
not the much larger shared-filesystem free space. No training or environment
probe was run on the login node; simulation jobs and environments are untouched.

All seven saved heads, the full-refit checkpoint, failed log, registration and
protocol are now backed up in a new isolated subdirectory of that owned M3W
root. All11files (5,090,052bytes) were read back and hashed exactly. No local
file was removed. Remote backup manifest SHA256:
`65ff07d6b35207fc2facd7c56e541ebaedfcacb8150e69f11fb4855fe2a453eb`.
This is checkpoint preservation, not remote training or a completed migration.
No new CREATE job was submitted by this recovery.

## Next Executable Work

Prepare a diskless, hash-bound G-only portable training pack and qualify the
independent M3W CPU environment on a compute allocation before migrating any
remaining fits. Check resumed optimization and numerical parity, not just Torch
import. Keep all18jobs, source roles, objectives, seeds and fourfold budgets.
Cross-platform exact prefix reproduction has not been tested and must not be
assumed. Do not borrow the simulation tree or its environment.

If local free space recovers above the existing reserve, the unchanged local
command is `scripts/run_m3w_feature_duration_probe.py train --resume` under
`.venv-pytorch/bin/python`; use a new resume log rather than overwriting the
failed log. Completed fits are reused, the seventh checkpoint resumes, and no
old optimizer state or observation is discarded. The scientific goal remains
active and unmet. This is the first resource interruption, not a repeated
hard-block verdict. Stage5C execution and SMC remain off.
