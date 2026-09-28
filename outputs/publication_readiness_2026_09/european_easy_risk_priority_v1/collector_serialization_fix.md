# Config-Receipt Serialization Check

The first real-pilot collection stopped at a receipt-hash assertion. Training
had already completed normally; no checkpoint, model or input was changed.
The collector compared the bytes of the compact local config with the bytes
of the semantically identical indented JSON written by the remote submitter.

- Registered compact config SHA256:
  `ebbebb68af44ec5ca1807291f34d1b05f3144e0da964cc23c2bb26ebd6dd3c6e`.
- Expected remote serialization SHA256:
  `40ecd4343e8e0e77ca7d79e65ec82836e221c830a717656fecfc0e2c84bec1ff`.
- The pilot receipt contains the latter value. Its registration SHA256 also
  exactly matches the local registered record.

Only unregistered collection/report adapters now compute the expected remote
serialization. Registered source/config bytes remain unchanged. A regression
test accepts formatting-only equivalence but rejects changed update counts.
Collection then verified both real checkpoint file hashes, matching samples,
100updates per arm, no unknown-label draws and the recorded risk-projection
bound. No failed training was hidden and no scientific rule was amended.

The 200-update pilot took12.9939seconds inside the runner and49seconds in Slurm
accounting, with peak Slurm batch RSS2,636,492KiB. The first-pair extrapolation
is6,783.80seconds, not a measured full-run time. This passes the existing
resource guard; full training was submitted once as37576457 and last observed
PENDING. Full training, paired replay and held efficacy remain incomplete.
