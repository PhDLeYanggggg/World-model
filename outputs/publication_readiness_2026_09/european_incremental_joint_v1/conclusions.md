# Joint Additions Do Not Yet Improve The Controller

I completed the registered same-predictor comparison: seven policies,36
source-role/seed/event groups,252 policy views and3,000 source-bootstrap resamples
per supported comparison. The frozen predictors and cost heads were reused after
hash verification; control inference and outcome evaluation were newly run.
No new neural network was trained in this experiment.

## What The Experiment Shows
Joint control does not establish an accuracy advantage over independent gain
ranking: all-ADE differences range from -0.012203% to +0.003400%, with no positive
or negative confidence intervals. Removing the pair-product terms gives a
similarly close result: joint-minus-unary ranges from -0.000553% to +0.001953%,
again with no positive interval. Better performance than hash-priority selection
is not sufficient: gain ranking is the stronger matched comparator.

The joint mechanism has little room to act under the frozen support gate.
There are13,824 repeated query/views, but only73 with feasible non-additive pair
support. Joint and unary choices differ at six query/views, covering only three
unique current queries. These counts cannot support a broad multi-agent claim.

Reducing the additions is not automatically safer. Relative to the frozen
full-add diagnostic, the half-count joint policy loses all-ADE in every view,
from -0.677313% to -0.008783%, with27 negative intervals. Its worst-locality
positive-easy mean degradation versus CV is2.254973%, exceeding the2% gate.
Full-add is1.394771% on this same population. This does not promote full-add:
it remains an opened-source diagnostic with residual hard failures and no
independently calibrated risk guarantee.

## Scope And Decision
The comparison covers6,116 unique indexed rows at1,152 fixed past-hash queries,
not the full318,969-row parent population. That difference explains why its
easy summaries must not be compared as if they were a retraining change to the
previous full-population0.0% summary. Twelve unique source localities recur in
four-locality readout groups; neither windows nor36 views are independent tests.

One hash-control solve failed its numerical certificate. The preregistered
fallback retained the independent choice for all matched controls in that query.
It was retained in the full comparison and marked uncertified, not hidden.

**No deployment promotion. Not yet a CCF-A submission candidate.** The specific
claim that this proximity-based joint addition mechanism improves the frozen
controller is unsupported. The next priority is independently tested incremental
risk ranking/calibration and a stronger same-predictor deferral comparison, not
increasing the joint penalty after inspecting these outcomes.

Deployment stays unchanged. No Stage5C, SMC, independent confirmation,
metric/seconds, physical-safety, human-gold, true3D or foundation-success claim.
