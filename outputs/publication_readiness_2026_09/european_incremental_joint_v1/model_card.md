# Frozen Incremental Joint Controller

Status: research diagnostic, not deployed. New controller decision experiment;
no new trajectory or cost-head fitting. Cached assets are the hash-verified
European incumbent-relative v1 forecasters and cost heads.

The existing incumbent's neural selections cannot be removed. Additional
choices come only from its frozen positive-gain, latest-motion and predicted-risk
support. Half-count gain ranking, hash priority, unary geometry and pairwise
geometry share cardinality and the reference predicted-harm cap. Geometry uses
current bounding-box widths and predicted12-step trajectories, never ground-truth
future trajectories. Domain, scene and coordinates are not mixed across units.

Inputs contain no future endpoint, future validity mask, central velocity or
held-out endpoint goals. Future labels are used only to score frozen decisions.
The original disjoint producer/controller/readout source roles and three seeds
remain intact. Opened-source development history is not erased by that split.

The pairwise method has no supported ADE gain over independent or unary
selection and fails the2% easy preservation criterion on the query subset.
Predicted proximity and predicted harm are not physical safety guarantees.
Unindexed agents without sufficient history are not jointly controlled.

No new deployment, independent confirmation, metric/seconds claim,
Stage5C execution, SMC, true3D or foundation-model status.
