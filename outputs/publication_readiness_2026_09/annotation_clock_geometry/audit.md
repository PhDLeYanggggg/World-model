# Annotation Clock and Geometry Evidence

Fresh local metadata inspection and coordinate replay, not a new model evaluation. The running v6 task and error units remain unchanged.

| Recording | Header FPS | Documented interval | Frame-step values | Clock conflict |
| --- | ---: | --- | --- | --- |
| seq_eth | 25.0 | 0.4 | ['6.0'] | True |
| seq_hotel | 25.0 | 0.4 | ['10.0'] | False |
| students03 | 25.0 | None | ['10.0', '20.0'] | False |

ETH and Hotel source info explicitly describes 0.4-second annotation intervals. However ETH obsmat uses six-frame steps while its AVI header declares 25 fps: 6/25 is 0.24, not 0.4. A source/encoding clock mapping is missing; do not select the more favorable interpretation. Hotel ten-frame steps are consistent with its documentation and header, but independent video/annotation synchronization is still not established.
Twelve intervals would be 4.8 seconds only conditional on the documented 0.4-second clock. Eight observed points would span seven intervals (2.8 seconds), not eight. Neither is promoted to a verified benchmark conversion. The current protocol keeps native observation steps and raw-frame offsets; raw50 effective seconds remain unverified.

Students03 has 21859 pixel rows and 21846 stored rows. All stored rows match unique frame/agent pixel keys; 13 pixel-only keys at frame5391 are listed in the JSON rather than silently discarded. Their omission reason is not established here.
The supplied H-old.txt reproduces stored coordinates with maximum error 7.0364419e-07; H.txt differs by up to 21.8099744 stored units. The current conversion script names H.txt, so that script/matrix pair does not reproduce this obsmat artifact. A future scene-image pipeline must bind the matching matrix explicitly. This establishes numerical lineage for H-old, not physical ground-plane accuracy or metric units.
The supplied px2ground.py uses forward differences for velocity. The experiment reader takes only frame, identity and position columns and recomputes causal finite differences; supplied velocity columns are not model inputs.
UCY source documentation describes interpolation between sparse spline annotations. Past-only reader access does not verify the causal construction of those annotation positions. A strict sensor-as-of claim remains unsupported.
Students01 lacks a locally matched homography/video pair here. Zara03 fitting remains a packaged 20-point population; this audit does not recover its continuous identities. Neither issue is silently repaired or used to redefine the live training experiment.
Homography presence, source-reported meters, verified geometric measurement and true 3D are different claims. No SDD timing or geometry conclusion follows from this audit. No metric/time performance claim, new deployment, Stage5C or SMC.
