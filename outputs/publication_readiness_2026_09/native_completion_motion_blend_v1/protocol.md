# Can Missing-Label-Aware Learning Recover Useful Pair Corrections?

The prior relative-loss experiment improved available pair steps
but did not improve the conservative full-grid result. Keep its
two source-separated frozen forecasters. Learn a past-only scalar
interpolation between individual-only (reference) and pair-trained
(proposal) predictions. A small shared MLP consumes observed speed
history and predicted path disagreement, not future labels or masks.
Both learned arms start at the identical half interpolation; retain
the fixed half interpolation and the unchanged individual path as
controls.

Use exactly the earlier 69,746 fitting rows, nine sources and
past-only same-query edge graph. Recompute the two frozen forecasts
only on these fitting inputs. For each seed, train both blend heads
for 900 updates with the same 64 individual and 32 edge draws per
batch. Both optimize available-step individual ADE and 0.25 times
observed pair error with the full twelve-step pair denominator.
The missing-label-aware arm alone adds maximum individual
disagreement from the reference on wholly unobserved rows and 0.25
times relative-path disagreement on unobserved pair steps. Both
debits use past scales; partial individual rows keep their
available-step denominator. The missing terms do not invent labels.

Freeze all development blend outputs before opening development
labels for this score. Keep the original gain recipients, genuine
query counts, CV fallback, observed labels and 2% risk rule. If
any of the fixed or fitted paths would be identity to CV within
one millionth of a pixel, all keep the original chosen path.
The primary comparison is missing-label-aware vs the frozen
individual arm for full-grid relative-path lower gain. Also
compare against the matched observed-only head and fixed half
blend, report conservative single-agent utility, easy risk and
original gain. Average three seeds before 3,000 paired source
draws. No validation or test selection.

This is a development diagnostic, not a revision to uncertainty
accounting or an independent result. Use arm64 CPU, two Torch
threads, zero loader workers and atomic checkpoints with heartbeat.
No external data access, promotion, Stage5C or SMC. Units remain
image pixels and native eight/twelve annotation steps.
