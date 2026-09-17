# Search-Support Follow-up, Not a Time-Offset Fit

The completed first correspondence diagnostic has 84 adjacent observed pairs per
source. All ETH displacements fit inside the fixed +/-24-pixel search square.
For Hotel, 56/84 later observed annotation positions are outside it. The 28
in-range Hotel pairs have mean errors 2.523 and 1.780 pixels for the fixed 15/31
templates, versus 32.391/26.518 on matched out-of-range pairs. These are descriptive
past-annotation diagnostics, not a forecast model or held-out result.

Repeat the same selected controls, direct frame-index convention, templates and
code with only search radius increased to 64 pixels. This is an adaptive diagnostic
to test a identified search-capacity limitation, not an independent experiment
or a time-alignment optimization. No future forecast labels or development scenes
are used. Preserve and report both radius settings. Do not select a radius from
forecast error, move the projected points or fit a temporal offset.

The larger search may exceed image boundaries; the existing matcher explicitly
returns missing support instead of padding. Report all requested pairs and missing
counts. Compare errors on jointly supported pairs as well as all matches. An
improvement only after dropping boundary pairs cannot certify all annotations.
Retain sources, outputs and failed/missing cases; do not overwrite the initial run.

No new training, physical clock claim, metric calibration, Stage5C or SMC.
