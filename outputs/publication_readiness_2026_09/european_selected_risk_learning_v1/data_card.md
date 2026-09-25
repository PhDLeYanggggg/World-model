# Data Card: Source-Only Cost Learning

## Material Passport
Cached_verified European detector-derived image-pixel tracks and forecast chains;
fresh risk-head training and fixed source-C evaluation. This is not SDD raw-t50
recertification, a metric/seconds benchmark, human-gold labeling or independent
confirmation. Main protocol: obs8/pred12 annotation steps at raw stride12.

The source cohort has 318,969 indexed rows across 12 localities. Its rosters
contain 116,823 / 23,762 / 178,384 rows. Source A fits the complete forecasting
chain; B supplies the cost-head training examples; C is the remaining roster.
All ordered A/B pairs and three seeds are retained. C excludes both fitted
chains for this readout but was historically opened development data.

Features are the fixed 383-dimensional causal bridge schema. Observed history,
geometry, two delivered forecasts, a causal CV rollout and policy bits enter
inference. Full and motion-only pairs are distinguished. Motion-only removes
neural trajectories and neural policy bits. All learned normalizers use B.
Future costs enter the B loss and C evaluation only, never feature construction.
No central velocity, future endpoint input or evaluation-endpoint goals.

Unknown labels are not zero errors. They remain in inference/query budgets and
action counts, but never enter training draws or supported error reductions.
The exact known support and IDs are bound in private receipts. Scene queries
group indexed agents by locality, recording and observed frame. They do not
claim to contain every real-world agent missed by the released detector.

Support bins are descriptive B-normalized feature-distance percentiles, not
an OOD guarantee or rejection policy. Easy events remain A-cut, positive-CV
events; their harm denominator remains delivered R error. Thresholds are fixed.

The six already-opened selection localities are not evaluated this round;
12 reserved calibration and six confirmation localities remain closed.
Overlapping windows, seeds and source-role settings are not independent samples.
Private arrays/checkpoints remain outside Git; public artifacts contain only
aggregate metrics, maps and checksums. Stage5C and SMC stay off.
