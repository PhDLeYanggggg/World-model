# Static Camera Projection Audit

## Material Passport

Fresh_run: enabled publisher control-point fits, leave-one-point checks and
reprojection against supplied polygons for all14 previously admitted fitting
cameras. Cached_verified: original fitting-camera identity audit. Not_run:
trajectory forecasting, policy scoring, raw-video resolution verification,
physical scale validation, new dynamics training or independent-role access.

QGIS sourceY is negated and point-file coordinates are rescaled using explicit
filename resolution to the polygon filename frame. OpenCV least squares uses
centered destination coordinates, no RANSAC or target-selected tuning. CRS
EPSG3857 is publisher metadata, not verified ground-truth metric accuracy.

| Camera | Enabled points | Point / polygon frame | Point inverse RMSE px | LOO inverse RMSE px (undefined) | Supplied polygon inverse RMSE px |
|---|---:|---|---:|---:|---:|
| 7 | 4 | [1280, 720] / [1280, 720] | 1.8231e-05 | not_estimable (4) | 4.95283 |
| 8 | 13 | [1920, 1080] / [1920, 1080] | 34.3912 | 49.8294 (0) | 17.9372 |
| 20 | 57 | [1920, 1080] / [1920, 1080] | 39.7095 | 44.6184 (0) | 28.5485 |
| 45 | 16 | [1280, 720] / [1280, 720] | 29.704 | 80.4123 (0) | 70.6125 |
| 48 | 8 | [1920, 1080] / [1920, 1080] | 11.6412 | 22.0407 (0) | 25.5315 |
| 67 | 13 | [1280, 720] / [1280, 720] | 13.7802 | 19.5974 (0) | 72.2915 |
| 68 | 5 | [1280, 720] / [1280, 720] | 18.3102 | 410.949 (0) | 39.5758 |
| 74 | 12 | [1920, 1080] / [1920, 1080] | 17.1076 | 24.8927 (0) | 26.3755 |
| 82 | 10 | [1920, 1080] / [1280, 720] | 18.7341 | 33.5127 (0) | 423.622 |
| 110 | 14 | [1920, 1080] / [1920, 1080] | 75.6612 | 112.068 (0) | 50.12 |
| 112 | 18 | [1280, 720] / [1280, 720] | 27.5252 | 33.2949 (0) | 37.8154 |
| 119 | 45 | [1920, 1080] / [1920, 1080] | 42.5287 | 50.685 (0) | 37.5478 |
| 124 | 16 | [1920, 1080] / [1920, 1080] | 27.1315 | 47.8157 (0) | 9.17006 |
| 126 | 19 | [2560, 1440] / [1920, 1080] | 52.3029 | 72.1995 (0) | 62.8744 |

These are internal static resource checks, not independent surveyed geometry.
The raw benchmark uses bounding-box midpoints; the publisher processing code
applies mapping to box bottoms. Raw recording frame dimensions are not supplied
by this audit. A point/polygon fit does not resolve that observation-convention
difference or authorize converting reported trajectory errors to metres.

Next: verify raw recording coordinate resolution and observation convention
before a separately registered projective forecast proposal. Do not silently
replace existing scene features, labels, normalizers or benchmarks. No new
forecast result, metric/effective-second, true-3D or foundation claim follows.

The publisher implementation documents resolution rescaling, sourceY sign and
box-bottom mapping: [primary source](https://github.com/kaktusracing/pedestrian_trajectories/blob/main/csv_functions_v2/calc_geo_df.py).
Its code was read, not executed or copied into the repository. Stage5C and SMC
remain off, and independent roles and deployment remain unchanged.
