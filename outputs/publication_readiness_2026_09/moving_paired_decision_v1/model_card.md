# Paired Intervention-Risk Heads

Intended use: controlled research on neural intervention gain and harm under
the original evaluation contract. The model neither produces new forecasts nor
provides a calibrated safety probability. No automatic deployment.

Architecture: separate64 risk and decision encoders, 64,260 parameters and492
causal inputs. Existing history, neighbor, frozen rollout and current-query
features only; prior latent columns remain masked. No future truth, future
validity, test normalization, central velocity or new endpoint goals at inference.

Training: fixed joint / paired_point / paired_rank objectives, three own-B
contexts, one producer seed, 2,000 updates per head. Cyclic useful/harmful pairs
stay inside queries and use only observed supported nonzero targets. Original
supervised anchors and unknown-label bounds remain unchanged. The paired loss
does not replace inference risk/support rules or change the zero threshold.

Evidence: consult completed fit receipts, README and verification, not this
architecture description. Same-data exact replay is engineering reproducibility,
not independent confirmation. The original24-group primary remains NA, not pass.
There is no identity exemption for missing reference/support or empty controls.

Native obs8/pred12 image-coordinate development only; not metric, elapsed
seconds, true3D, foundation or human gold. Stage5C/SMC disabled. Private data
and checkpoints are not distributed through the public repository.
