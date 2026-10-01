# Storage Amendment, Before Scheduler Submission

The first frozen-input stream stopped when its cumulative compressed packet
size exceeded the registered 512 MiB temporary remote storage cap. No compute
job was submitted. This is a storage estimate failure, not a scientific result.
The completed remote packets and all local source/partial-transfer outputs stay
unchanged. This amendment raises only the remote packet cap to 2 GiB. Personal
quota remains unknown; any quota or write failure stops the stream. Nothing is
deleted, and no local array cache or simulation-project change is allowed.

The exporter resumes by recomputing identical frozen packets and checking the
existing remote SHA-256 before reuse. Scientific registration, input precision,
cost heads, calibrators, masks, 2% budget, source screens, roles and exact-replay
requirements are unchanged. All 216 views remain required, with no selection
based on readouts. Scheduler allocation stays 4 CPUs, 8 GiB, one hour, one job.
Independent confirmation is closed; Stage5C and SMC remain off.
