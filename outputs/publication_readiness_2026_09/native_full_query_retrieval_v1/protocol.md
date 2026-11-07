# Full-Query Recipient Retrieval Under A Fixed Shortlist

Can a past-only learner recover the added opportunity from allowing
previously excluded recipients, without increasing the query switch cap?
On the exposed Moving A data, use the fixed path bank plus the same
OLS4 common translation for both arms. Train two ridge heads on
identical past-motion, neighbor, and rollout-disagreement inputs.
Exclude the entire scoring source from fitting. Give sources equal
mass, queries equal mass within sources, and rows equal mass within
queries. Use three frozen forecast seeds, not three optimization seeds
for the deterministic heads.

The control ranks eight eligible recipients by learned normalized best
CV gain. The alternative uses the same first four (round up when fewer
than eight are available) and fills the remainder by learned local
opportunity. The second target is the highest normalized gain in a
locally feasible singleton or pair containing that person, using original
risk and reference conditions restricted to the query. The target
uses labels only in fitting; it is not a recording-level certificate.
Known rows use their best legal forecast. Wholly missing rows cannot
improve local CV gain or supply easy support, so do not enter target
pairs; they are still present at inference and in the evaluation.

Primary: positive conservative OLS4 capacity of each retrieved shortlist,
using the same original cap, all paths on its recipients, and the same
per-recording 2% checks. Retain solver timeouts and report 3,000 paired
source bootstrap draws, averaging forecast seeds. Secondary: original
joint decoder and gate applied to each shortlist with the same cost
fit, plus the original unshortlisted gated controller as a reference.
Matched proposals have identical counts; gated policies need not.
Evaluate CV/OLS4 utility, hard/easy summaries in the scorer and actual
recording passes. No recalibration or use of external data.
