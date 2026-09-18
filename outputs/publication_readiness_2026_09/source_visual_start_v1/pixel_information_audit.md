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

## Source Box Extent

The 36,234 unique past SDD crops have median annotation-box width/height 10.915/12.657 model pixels.
The smaller axis is below four model pixels in 924 crops and below eight in 11,417 crops.
An axis extends outside the 32-pixel crop in 27 cases.
These are mapped annotation extents, not visible-body masks or a certified posture-resolution threshold.
Main-site box-size audit is not_run because the current main cache has no verified box extents.

This audit neither filters the registered population nor changes training, thresholds or evaluation roles.
