# Opened European Source Query Population

Source: the existing verified EuropeanSquares released detector-track asset
chain. Image-pixel coordinates; obs8/pred12 native steps under the frozen source
sampling protocol. Effective seconds and metric calibration are not established
by this study. Detector-track labels are not human gold.

Population:6,116 unique indexed agent/query rows across1,152 current queries,
96 pre-existing hash-selected queries per source locality and12 unique localities.
Three seeds, two event targets and six producer/controller/readout rotations
reuse these rows. Their13,824 query/views and252 policy views are not independent
sample counts. Each outcome comparison averages four readout localities equally.

Query hashes depend only on locality, recording and current frame. All indexed
eight-history targets at each selected query are kept, including those with no
future label. Partially known trajectories contribute masked ADE, not fabricated
points; complete-window ADE is separate. Endpoint FDE requires the actual final
label. Missing denominators stay undefined. Full per-view counts and tail/error
summaries are in groups/*.json.

Neighbor geometry in this controller includes indexed targets only. The graph
does not represent every visible person and does not invent forecasts for short
or missing histories. Future labels and future validity are absent from its
decision interface. No test endpoint is used to make goals. No normalization is
fitted to readout sources; the prior controller-training cost scale is reused.

The parent includes318,969 rows, but this experiment is explicitly a restricted
query study. Its safety summaries are not interchangeable with the full parent
population. Reserved model-selection, calibration and confirmation data remain
unopened. This round adds no raw data or trained latent cache to GitHub.
