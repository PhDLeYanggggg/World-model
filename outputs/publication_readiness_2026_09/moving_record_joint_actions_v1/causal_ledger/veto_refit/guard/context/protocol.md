# Context-Matched Transfer Of The Radius Guard

Reuse the exact 42 frozen parent cost fits, residual fits, source roles
and radius threshold grid. The learned harm score did not reliably
improve the radius control, so the new factor is calibration source
matching, not another learned guard. Keep all three parent calibration
sources distinct from all ten fitting sources and the one held source.

Describe each recording using only the first current query's eight
observed steps: crowd size, speed relative to current neighbor spacing,
path straightness, still-agent fraction and whether neighbor spacing
is supported. Scale those features with the interquartile range of
the ten fitting sources' first-query profiles (floor 0.10), using
Euclidean distance. No later query, coordinates from another time,
label or outcome informs the match. The threshold is picked once
at the first query and held for the recording.

Nearest chooses one recording from each of the three fixed calibration
sources. Context-blind chooses one per source by stable ID hash;
farthest is a same-count negative control. Each picks the positive
conservative utility threshold that preserves the parent calibration
criteria on all three selected calibration recordings, or uses no
reopening. This matches calibration counts and source roles; compared
with the pooled parent policy, it changes the recording coverage and
can reduce representativeness. Calibration labels are allowed only
on those fixed inner sources. The outer labels are never used to
match or choose a threshold.

Original, unguarded and pooled-radius controls replay exactly. All
arms match the same 6,344 switches query by query and maintain causal
predicted balances. Primary nearest minus context-blind direct
completion lower; secondary nearest minus farthest and pooled. Keep
all 102 recording views and the original observed 2% all/easy,
signed easy-net and support checks. Use 3,000 paired source bootstrap
draws after seed averaging. This is empirical source transfer on
exposed development, not formal calibration coverage or independent
evidence. No new forecaster training, deployment, Stage5C, or SMC.
