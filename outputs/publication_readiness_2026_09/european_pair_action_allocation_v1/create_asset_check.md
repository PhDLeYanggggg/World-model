# CREATE Read-Only Asset And Scheduler Check

Fresh authenticated observations on1November2026. These infrastructure checks
are separate from the local scientific result and do not promote a model.

## Observations

- Queue check completed13:33:50UTC,exit0. No M3W-named rows occur in the returned
  snapshot. This is not proof of absence of historical or differently named jobs.
- Owned M3W metadata check completed13:36:49UTC,exit0. Its owner marker verifies
  the authorized root. Seventeen direct project directories and six named small
  JSON receipts were inspected. No remote home or simulation tree was scanned.
- Accounting for the six job IDs in those receipts returns COMPLETED/0:0 for
  all six:historical CPU runtime,diagnostic and selected-pool/calibration jobs.
  These are dated scheduler observations,not inference from an SSH login.
- Observation availability for those six jobs is restored. This does not
  revalidate numerical science,all historical jobs,three forecast-producer
  seeds or the post-migration GPU/Torch environment.

Only named metadata files smaller than2MiB and their hashes were read. No
input,label or checkpoint contents were read. Nothing was submitted,cancelled,
restarted,installed,uploaded,deleted or changed remotely. Existing credentials
and the protected simulation work were untouched.

## Local Execution Decision

The decoder readout actually finishes locally in2563.680355s at10,848,567,296B
peakRSS,native arm64 CPU4/interop1/workers0. It fits the48GB machine and12h
healthy-run bound. No CREATE resource is needed to duplicate it. Larger
forecast-seed training would need current quota checks and a scheduled
post-migration runtime validation first.

## Private Evidence

Detailed observations remain in the ignored local experiment directory,not the
public or anonymous reproduction package. Account,key,internal host,protected
workload and author identifiers are omitted.

Queue receipt SHA256:
`09f6339291d6b123d8bdc246ef1f5bffb0642cfa6eacb309e7b7a509dbca94c0`

Owned metadata/accounting receipt SHA256:
`dec58e7969fb796766e430897d190a6639e86c8656b48740802eea0658035535`

The bounded read-only helper is scripts/audit_m3w_owned_create_metadata.py.
It refuses to overwrite its dated receipt. No scientific gate,independent
role,Stage5C or SMC status changes.
