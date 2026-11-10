# Choose from three frozen candidates at the same intervention count

Use the existing additive two-recipient anchors and the first three
nonzero rotations in the old ego score order. The top-one choice,
query, recipients, path displacement and original 2% risk rule
remain fixed. Each policy may choose a different candidate on the
shortlist or keep the incumbent when not selected. No label can
change the shortlist.

On donors, generate anchors with the nested additive chooser and
nominees with the nested ego chooser. The source-excluded upstream
forecaster, chooser and angle model exclude that donor and both
outer locations. Use the exact one-recording roster and retain
three alternatives per recording for both heads. Use the same
architecture, 200 updates, batch 256 and three seeds.

The strict head predicts the previous strict criterion. The utility
head also predicts original risk, risk plus both references and
positive incremental gain, gain and harm separately, and relative
motion. Its fixed score is the past-only displacement envelope
times gain minus harm plus half relative gain, minus 0.5 times
risk probability deficit and 0.25 times release probability
deficit. Trajectory harm charges unobserved steps their maximum
possible change. OLS4 agreement is an untrained choice control.

Report fixed-mask action choice, top-one reranking, and top-three
choice plus reranking. Use the same 526 and rounded 15% recording
budgets, and also separate calibration by the opposite location
using the original risk, dual-reference and positive gain checks.
Do not change a threshold after held scoring. Compare directly
against the fixed top one and the same head on top one. Report
paired conservative utility, observed endpoint gain, relative
motion, and location-resampled intervals. OpenDD is development;
no new external outcome access or release, Stage5C or SMC.
