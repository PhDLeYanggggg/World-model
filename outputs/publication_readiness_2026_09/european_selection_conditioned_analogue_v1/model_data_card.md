# Fixed-Policy Donor Conditioning: Diagnostic Card

Status: fresh72-view fitting repair attempt; magnitude gate failed. No new neural
weights, trajectory rollout, deployed selection rule or independent test result.

## Inputs and Reference Boundary

Frozen380-dimensional causal motion/neighbor/baseline features and original
causal policy mask. The mask contains no query future endpoint, known-future
availability, central velocity or test-goal construction. The features retain
log(observed scale); complete coordinate invariance is not claimed.

Four same-context inner-fit sources are pooled; the whole source-qualified
query recording is excluded from reference normalization and labels. Full-bank
normalization includes unknown feature rows, unchanged between arms. Selected
supervised donors must pass the same original policy; labels are used only on
known donors. Unknown queries stay in prediction/support accounting. The
upstream predictors, preprocessing and original policy were already trained;
this is not whole-pipeline OOF.

Uniform32-neighbor means predict H/envelope and easy-H/envelope, multiplied by
the query envelope. B/R/easy-R are copied from the frozen original. Mean and
within-recording paired-label shuffle are controls. Full-bank numerical controls
are exact. Insufficient selected donors would remain unsupported; none occurred.

## Meaning of the Diagnostics

Easy-harm event means easy-H>0, not easy membership alone. Event probability
and positive severity are factorizations of the donor mean, not newly trained
classifiers. Query future outcomes enter scoring only. Event-positive severity
error is support-conditioned, not an inference feature or unconditional metric.
Signed-risk MSE, normalized harm SSE, event Brier and total harm mass have
different weights/denominators and are not ADE/FDE improvements.

No deployment recommendation. Selection conditioning reduces error versus the
full donor bank but remains worse than the original head; favorable event
frequency does not establish risk magnitude. Rare events, unknown outcomes,
recording dependence, uneven source sizes, source020 leverage and upstream
exposure limit conclusions. No confirmatory CI or p-value is reported.

Protocol: obs8/pred12 rawstride12, pixel/local detector-silver, not metric,
seconds, human gold, physical safety, true3D or foundation. Risk budget remains
2% positive selected harm/reference. Stage5C and SMC are off. Cached artifacts
are identified separately from fresh calculations and not_run evaluation.
