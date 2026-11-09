# OpenDD two-edit anchored ranking

Development only. Keep all seven exposed OpenDD locations, three existing
seeds, six ordered calibration choices per held location, 8 observed and 12
predicted steps, two edits, frozen forecast bank, and the original 2% rules.
No Songdo or DroneCrowd confirmation data are read.

In each fixed query, take the frozen additive proposal as the anchor and
rank alternatives preserving at least one of its recipients. Exclude the
exact anchor from the alternatives, never expand the bank. On donors, use a
seeded random legal two-edit anchor rather than a model fitted on those
donor outcomes. Treat this anchor distribution difference as a limitation.

Train two equally sized heads on identical donor pairs and updates. Both
learn the original risk and trajectory targets. Only the relative head
learns the conservative trajectory and relative-motion gains against the
anchor. Compare each against frozen additive and unrestricted joint
proposals at the same edit count. Fit on the five other locations; choose
thresholds on the designated calibration location only. The calibration
prefix requires the original dual and 2% checks, no trajectory loss against
additive, and strictly positive conservative relative-motion gain. No
prefix means no reranking. At evaluation, only the frozen prefix replaces
additive; this is a development comparison, not certification of the
additive policy or the replacement as a deployable system.

Report candidate and admitted counts, joint criterion pass counts, and
paired conservative ADE and relative-motion contrasts. Resample seven
location means with the existing 3,000 draws, and retain per-seed readouts.
Never treat repeated recording views as independent. A result on these
exposed locations cannot replace the sealed independent evaluation.
