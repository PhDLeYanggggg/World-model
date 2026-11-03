# Lossless Completed-Pilot Storage

This is a fresh engineering operation, not new training or a scientific result.
All18primary models have completed and their original paths/bytes remain intact.
Original held inference continues asPID91599,independently of this operation.

Four completed100-update pilot copies originally total48,838,756bytes. Their
scientific resume/direct states agree except elapsed time,but raw serialization
layouts differ,so they were not treated as byte-identical files. A local lossless
XZ archive retains all four original byte streams in14,842,420bytes,with
33,996,336fewer logical bytes than the standalone copies. This is not a claim
about total filesystem free-space change while other processes run.

ArchiveSHA256:
`f295eb82f115e04ce155f3f60994d280a0e5a4544ada3b408e3c4f2e8afb8c0c`.
Each member's original path,size and SHA are in
[pilot_storage_archive.json](pilot_storage_archive.json).

Before compaction,all four archive members were read and matched byte-for-byte
by SHA against their original files. Before removing standalone copies,the
archive and manifest were synced and all existing originals were checked again.
A separate verify invocation passes after compaction. All18primary-checkpoint
hashes are freshly checked afterward and unchanged. Two synthetic archive,
round-trip restoration and changed-file protection checks pass0.12s. Build,
compaction and separate verification invocations each exit0.

Only these experiment-owned completed pilot copies move into the archive.
No raw annotations,other old data,source/protocol,primary checkpoint,trajectory
prediction or full-refit file is removed or reserialized. No loss,normalizer,
source role,seed,budget,threshold or risk definition changes. The original
pilot receipts and primary prefix-import lineage remain unchanged.

The private archive is
`data/stage_cvpr2027_experiments/moving_source_forecast_supported_v1/pilot_verified.tar.xz`.
Full4krefits start from seed and do not need standalone100-update pilot files.
To inspect the preserved byte streams or restore their original paths:

```sh
.venv-pytorch/bin/python scripts/archive_m3w_moving_pilot.py verify
.venv-pytorch/bin/python scripts/archive_m3w_moving_pilot.py restore
```

Restoration checks the original10GiBreserve plus missing raw-file space and
refuses to overwrite changed existing files. Restoring is a storage operation,
not a repeat of the immutable completed pilot phase or a new model result.
The archive stays private and is not committed to Git. Stage5C/SMCremain off.
