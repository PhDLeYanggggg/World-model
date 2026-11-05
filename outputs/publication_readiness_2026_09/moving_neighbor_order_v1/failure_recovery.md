# Metric Implementation Repair Before Readout

The first diagnostic process (PID23437) exited1 during the first context's
identity-view check, before saving any analysis or complete-context metrics.
Its new helper averaged nonzero channel costs, whereas the existing joint
decision diagnostic sums channel magnitudes and also counts zero-target
rejections. The identity value0.005704 differed from the frozen0.006460.

The repair removes the duplicate formula and reuses the existing
`m3w_decision_semantics.diagnose` implementation. A two-positive/two-negative,
zero-target and unknown-label regression case checks the exact original
semantics. The tiny synthetic preprocessing fixture was also made consistent
with the actual clipped inference input contract.

The initial registration is retained; `registration_corrected.json` binds the
corrected implementation to it. No trained model, prior result, label, split,
prospective screen, original2% rule or held data changes. This is an engineering
repair, not positive scientific evidence. The real diagnostic must rerun and
match the original identity metrics before any result can be reported.
