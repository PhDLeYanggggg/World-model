# Align the original frame, not the forecast outcome

Register the winter DJI_0932 raw frames to the municipal 2020 orthophoto.
Do not read positions from the trajectory release to fit the transform.
Use only the selected frame's pixels and the static map. Withhold four
named crosswalk patches (48-pixel exclusion radius, at 1280 x 720).
Use mutual SIFT matches outside those patches and a robust homography
with a three-map-pixel residual cutoff. Independently check the held
markings with local edge-template translations.

Frames are 0, every 120 through 1200, and 1219. At a usable frame,
require an inlier hull covering at least 18% of the frame, at least
three occupied quadrants, at least 8 inliers, map inlier p95 at most
3 pixels, and at least three held patches with peak above 0.30 and
displacement at most 6 map pixels. Check but do not refit the thresholds.
Record camera drift using the frame center, expressed in the supplied
map grid, not as a verified physical displacement.

If initial geometry passes, run a predeclared current-frame alignment
check at raw indices 0, 4, ..., 28. Reuse the earlier frozen COCO detector
and causal association, project its centers, then compare to labels at
the same timestamp only on the evaluation side. A fixed three-unit
radius defines association candidates. Report label and detector
coverage, distance, and track-ID consistency. These are calibration
diagnostics on a fixed early prefix only. No future forecast error,
test role assignment, or change to the existing evaluation.

The orthophoto request is UTM 17N WGS84 (EPSG:32617); the data provider
documents UTM zone but its precise export datum and object anchor are
not independently verified. All comparison distances are map-grid
units; alignment uncertainty is not a calibrated metric benchmark.
