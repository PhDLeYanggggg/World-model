# Readout I/O Repair, No Scientific Change

The first local independent-verification attempt was interrupted deliberately
with SIGINT (exit130), not reported as successful. Its process remained CPU-active;
this was not an OpenMP/SHM hang. The captured stack identified repeated
`NpzFile.__getitem__` inside the scalar full-reference sum: every row access
decompressed the entire target member again. No numeric result files had been
produced. The completed CREATE job37714473, inputs and outputs were not modified,
cancelled or resubmitted.

Reader v2 materializes each allowed NPZ member exactly once per packet in RAM,
with a512MiB decoded-packet bound, then runs the SAME inference, scalar sums and
bootstrap comparisons. The128MiB compressed-input/output bound and10GiB local
numeric-cache reserve remain. A regression test counts exactly one decompression
per member and compares materialized-versus-original dictionary results exactly.
No smaller sample, changed risk definition, new tolerance, altered calibration,
target tuning or transfer evaluation is introduced.

The v1 remote helper also imported Torch transitively although its verification
arithmetic did not use it. The prior no-Torch import test covered module loading,
not that helper's runtime path. V2 uses a bounded standard-library read-only SSH
call with the same hash-verified handoff and strict arguments; it asserts no
Torch import through the entire verification run. This is runtime isolation,
not a new neural-training result.

The original readout registration is preserved. Its implementation is recoverable
at commit c6f421a7. A separately frozen v2 readout registration binds this repair;
the original scientific experiment registration and all completed calibrations
remain unchanged. No fresh scientific hypothesis or preregistration claim is
made for this post-completion verification repair.
