# Streamed Installation Amendment

The initial installation connection reset before the experiment directory was
created. A fresh read-only SSH query returned `root_exists=false`. There was no
new pilot submission, fit or result to accept or repeat.

At23:25 UTC on28 October, a controlled read-only400,000-byte argument probe
also reset the connection (exit255), whereas the same400,000bytes through stdin
were received exactly with exit0. The registered JSON is508,992bytes. This
supports a large remote-command transport failure, not a PyTorch training hang
or missing general SSH access. The precise server-side size limit is unknown.

The amended installer sends JSON and its code archive through stdin. The
original ownership, byte-hash, immutable-write and path checks are unchanged.
Original training registration, numerical code, models, objectives, sampling,
splits and gates remain byte-identical. No new credentials, MFA approval,
network settings or other project's files are changed.

Use `python -m scripts.submit_m3w_task_risk_stream pilot` and, after successful
pilot collection, the same command with `train`. Observation and collection use
`scripts.manage_m3w_task_risk`. Explicit resume uses the streamed entry with
`resume --phase train`, only after a verified terminal failure. Ambiguous
submission never permits an automatic duplicate.
