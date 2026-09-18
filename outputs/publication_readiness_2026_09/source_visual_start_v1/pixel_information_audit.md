# Past Pixel Support and Temporal Variation

Descriptive input audit only. No labels or model scores enter these statistics.
Crops reduce a 96 x 96 source-pixel region to 32 x 32; physical scale differs by camera.

| Population | Rows | Supported adjacent pairs | Identical pairs | Entirely identical histories | Mean absolute pixel change (0-255) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETH | 81 | 567 | 0 | 0 | 2.472233 |
| Hotel | 284 | 1988 | 0 | 0 | 2.636930 |
| SDD_train | 22374 | 156618 | 18 | 0 | 1.911331 |

Only pixels supported at both adjacent times are compared. Missing pairs are not treated as unchanged.
Nonzero RGB change does not prove informative body-state visibility or correct intent labels.
Compression, lighting, other agents and crop registration can also change pixels. No physical time or scale equivalence is asserted.
This audit neither filters the registered population nor changes training, thresholds or evaluation roles.
