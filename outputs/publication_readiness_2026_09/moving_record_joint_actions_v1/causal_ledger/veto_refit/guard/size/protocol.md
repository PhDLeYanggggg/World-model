# Separate Calibration Sample Size From Additional Risk Constraints

Use the exposed Moving A roster with the same three frozen forecaster
banks and 8 observed / 12 future annotation steps in image coordinates.
For each held source, use the next three and the following three source
IDs in the circular sorted roster as disjoint calibration groups A and B.
The other seven sources fit all costs and residuals. Within fitting,
each donor proposal excludes its own source. Neither held nor any
calibration source fits the predictor. This compares calibration size
at a **fixed seven-source fit**, not to the earlier ten-source fit.

Reuse the frozen normalized-radius threshold grid and original policy
rules. Besides the no-reopening base and unguarded reopening, select
thresholds using A alone, B alone, all six for utility with A-only
preservation, all six for utility with B-only preservation, and all
six for both. Every choice maximizes positive paired completion lower
relative to the same base among thresholds that satisfy its assigned
calibration preservation criteria, or falls back. The two hybrid arms
separate additional utility information from adding constraints.
No radius threshold, model or setting is tuned on the held source.

All arms share the exact same query counts and unchanged original
observed 2% all/easy, easy-net and support rules. Different parent
study counts are not used as matched controls. Primary: all-six vs
the mean of the two paired three-source contrasts on the same held
population. The mean is a statistical contrast, not a synthesized
policy. Secondary: pool only utility versus matching three-source
control, pool constraints versus matching pooled-utility control,
fall-back frequency, disagreement between A/B, and observed original
risk. Keep all 102 recording views; 3,000 paired bootstrap draws by
source after averaging the three forecaster seeds. Calibration groups
are source-disjoint within each fold but reused across folds. This
does not certify formal risk control or independent confirmation.

No new neural forecaster training, external outcomes, deployment,
Stage5C or SMC. All outer choices freeze before their readout.
