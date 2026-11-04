# Observed Interruption And Same-Binding Resume

At22:21UTC on November4, fresh OSprocess inspection found PID89603 absent.
The original tool session34785was also unavailable. A process inventory found
no other instance of this study running. This is positive stopped-process
evidence, not an inference from an observation timeout. No remote duplicate
was submitted.

Last saved heartbeat:14:45:26UTC,6440.0074s of that process's monotonic runtime.
There are71complete2000update fit receipts (142000saved primary updates).
The next actual checkpoint is eqmotion_fold1_seed43_critic2_part2_replica1:
step100/2000, costseed1052, with optimizer, sampler, TorchRNGand own-source
normalization preserved. It was loaded successfully before recovery.

There is no terminal exit receipt or captured cause. I do not classify this as
an OpenMP/SHMfailure, out-of-memory failure, completed training, or a known
host shutdown. The later wall-clock gap is not training compute time.

Resume the unchanged registered train phase with `--resume`. Complete weights
are revalidated, not retrained. The partial head continues from its own saved
state; the remaining heads retain their full2000update budgets. Do not change
roles, losses, normalizers, thresholds or evaluation rules. No Vscore or new
Hreadout has been made. All108models must freeze before nested validation.

If the eventual training-execution receipt comes from the resumed process,
its elapsed time describes that terminal attempt, not the whole interrupted
run. Sum saved per-head fitting times separately; unknown unsaved work and
the unobserved shutdown interval must not be fabricated.
