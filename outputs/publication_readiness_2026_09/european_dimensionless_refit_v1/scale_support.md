# Observed Scale Support

All 318,969 source query histories were checked; no future labels or forecast errors were read.
Extent is the maximum past ego/valid-neighbor displacement from the current ego, in image-local units.
The inherited conditioning scale is max(extent,1). This is not a verified physical scale.

- Extent below1 (clamp strictly active): 700.
- Extent exactly1: 0.
- Extent at least4 (supports all0.25/0.5/2/4 probes): 317,587.
- Zero extent: 0.
- Quantiles [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]: [0.07622020691633224, 32.67548858642578, 100.86004486083985, 307.7127685546875, 693.1744506835938, 1080.1398193359375, 1327.90087890625].

A successful equivariance check on unclamped histories does not cover arbitrary unit changes
that cross the clamp, nor prove predictive generalization. No rows were filtered from training or scoring.
