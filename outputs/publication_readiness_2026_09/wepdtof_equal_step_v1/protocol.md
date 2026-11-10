# WEPDTOF: equal movement at every forecast step

Use only the 15 previously scored clips from 13 sources and the
three seeds. For each joint-versus-control comparison, verify the
original per-query edit counts and each edit's peak departure
from CV. Retain the selected recipients and directions. At every
step assign all edited paths the smallest original departure
from CV across both arms. Also keep the original and a control
that uniformly shrinks every edit to the same smaller peak.
All three variants keep the same roster. If the common profile
collapses to zero, omit that query from all three variants and
report the omission. Zero-edit views do not become usable contrasts.

Freeze the geometry from the verified forecast bank without reading
targets. On the existing outcomes, reproduce the original contrast
before scoring the movement controls. Use query-equal weighting,
the original observed CV denominator on the common roster, and the
original validity masks. Report direct observed/lower/upper utility
and relative-position bounds; report their denominators separately.
Average over clips and seeds within source, then use 3,000 paired
source resamples. Unavailable source contrasts stay unavailable.

This is a post-hoc sensitivity of the scored WEPDTOF test, not a
new threshold, a release candidate, or another independent test.
No new external labels, fitting, or changes to the original 2% rule.
