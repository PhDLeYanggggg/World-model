# Local Readout and Pending Remote Replication

The local memory-only reconstruction completed in 187.98 seconds with 7.492 GiB
peak RSS and zero new row-cache bytes. All 288 compressed packets match the
committed CREATE input hashes. Every diagnostic packet replays exactly locally;
64,512 native accounting checks, 672 summary reductions and 1,728 risk agreements
with the frozen parent readout pass. This is not a raw-video reconstruction.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/replay_m3w_boundary_local_inputs.py --register
# Registration must be committed before the first readout.
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/replay_m3w_boundary_local_inputs.py
```

Existing completed receipts are immutable. Do not delete them to rerun. The
receipt already includes exact per-packet numerical replay; verify its input,
source and output hashes instead of relabeling cached results as fresh.

CREATE job 37602475 is retained. It was observed pending after its walltime was
reduced from 120 to 30 minutes, with all 288 groups and the full replay unchanged.
A later connection check timed out; the remote job is not declared stopped.
The SSH input reader exited before reading any packet. The local reconstruction
avoids the unavailable transport and does not cancel or duplicate the remote job.

When access returns, inspect the same job. Collect only after COMPLETED/0:0 and
compare the complete remote and local summaries, ignoring only execution-origin
text. Permit documented numerical roundoff, not percent-level discrepancies.
Any mismatch blocks a cross-platform reproducibility claim.

32 scoped tests pass in the current diagnostic and parent modules. Full legacy
tests, new model training, cold raw rebuild and independent confirmation are
not_run. No simulation project or unrelated staged files are modified. No raw
data, row packets, checkpoint or credentials are committed.
