# Resource And Execution Record

The input caches are local and were hash-verified, so small CPU arithmetic is
appropriate. No GPU or new neural fitting was needed. Native arm64 Python,
CPU threads 4, interop 1 and workers 0; the actual diagnostic ran, not merely
a runtime import. About 20 GiB was available at start, above the 10 GiB reserve.
Observed full-run PID13955 used about 10 GiB RSS and sustained CPU activity.

Registration06511693 was pushed before the first pilot. PID13710 completed
one fitting view in14.846347 seconds including source loading after ancestry
checks. PID13955 completed all144 views in201.085832 seconds with the same
timing scope. These are diagnostic compute times, not training times or
end-to-end turnaround. Every view has an immutable result and checksum receipt;
the shared lock and resume path protect completed work.

The first figure invocation occurred before the asynchronous report process
finished and correctly failed on a missing CSV. After report completion was
confirmed, the figure was rerun successfully and visually checked. No input,
model, scientific configuration or numeric result changed. The reproduction
script waits for report completion before plotting.

CREATE was queried read-only at2026-09-26 20:00:24-20:00:26UTC,return code0.
Receipt SHA-256:
7984d619ca760521d679e0c15742dc087173bb7c318d4d5a07ad6023eb3ba011.
No remote job was submitted, cancelled or modified. Private SSH receipts,
cached scores, source data and per-view detailed intermediates stay out of Git.
No unrelated staged files are included. Fresh expanded checks passed51 tests
in seven scoped files; the complete legacy test suite was not rerun.
