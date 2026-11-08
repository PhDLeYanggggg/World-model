# Readout-only repair

The original train command completed all fits and wrote its frozen
weights. The original score command failed at JSON encoding after
numerical evaluation, because the source names and mixture-weight
summaries were arrays. No outcome artifact was written.

The readout runner calls the original scoring function unchanged and
converts arrays and scalar objects to JSON values at the final boundary.
It requires the original forecast freeze. No fits are repeated, no
scientific settings change, and no comparison is added or dropped.
