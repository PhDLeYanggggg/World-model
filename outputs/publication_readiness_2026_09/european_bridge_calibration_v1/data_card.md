# Data Card: Aligned Bridge Calibration

## Scope and Roles
This is an opened-development study, not an independent benchmark result.
Detector-derived European top-down tracks use image-pixel coordinates, eight
observed and twelve predicted annotation steps at raw stride 12. No seconds,
metric calibration, human-gold labels, true 3D or foundation-model claim follows.

The 12 source localities are partitioned into three four-locality rosters:
074/082/112/126; 007/110/119/124; and 008/020/048/067. For each ordered pair,
A produces the complete forecasting/floor chain and B fits utility/risk scoring.
C is the remaining roster and fits the calibration map. All three roles were
historically opened for development; the exclusion is from these fitted chains,
not a claim that the research never encountered C.

The readout reuses six opened selection localities: 087/092/093/103/104/125.
There are 28 recordings, 38,102 indexed targets and 7,087 scene queries under
the registered sampling cap. Label support is 21,434 complete, 15,820 partial
and 848 unknown futures; 37,254 rows support ADE and 27,694 support endpoint error.
These are not 38,102 independent scenes. Bootstrap units are the six localities.

The separate 12 reserved calibration and six confirmation localities remain
closed. DroneCrowd confirmation is unchanged. No independent role is silently
reassigned, and old test-selected Stage35/37/43/44 figures remain exploratory.

## Input and Target Separation
Input features contain observed history, causal geometry and the two fixed
delivered forecasts. Unknown-future rows remain in inference and action counts.
Future coordinates/masks enter source-C cost calibration and later evaluation
only. They do not enter model inputs or the selection decision function.
No central velocity or evaluation-endpoint goals are used.

Easy membership is defined by positive CV error below the A-derived easy cut.
The positive-harm denominator is instead the delivered reference policy error.
Missing event support is undefined and cannot pass a risk check. Public data
are aggregate maps, metrics and hashes only; tracks, feature arrays, model
weights and per-row labels remain outside Git.
