# A second published architecture on the fixed OpenDD vehicle task

Use the MIT-licensed Social-STGCNN core at the author's pinned CVPR
revision. This is a deterministic adaptation, not the original
pedestrian benchmark or best-of-20 sampling. Replace its five-parameter
Gaussian head with two mean increments. Keep one graph block and five
extrapolator blocks and preserve the author's time/feature layout.
Use a batched form of the same graph convolution, checked against
the original single-graph path.

Use the same OpenDD queries, focal comoving frame, nearest three
observed neighbors, CV cap, completion loss and sampling budget as
the EqMotion vehicle control. Build normalized Laplacians from
observed, scale-normalized relative increments; masked padding
has no graph edge. Compare peers against ego only. The network
never receives future labels or masks, which are loss/readout only.

Fit the seven leave-location-out folds with seeds 17, 29 and 43.
Use 400 AdamW updates per fit, batch 128, four native CPU threads,
no data workers, and checkpoints every 100 updates. Select no
hyperparameters on held outcomes. Freeze all 42 weights before
scoring. These locations are development, not new confirmation.

Reproduce CV, OLS4 and the incumbent on the same cohort. Keep
the prior 2% recording tests, donor-only easy/hard cutoffs and
3,000 paired location resamples. Compare available ADE/FDE, p95
and lower paired gain versus the vehicle incumbent and versus
the ego-only ablation. Report negative results and leave every
external confirmation dataset and release decision unchanged.
