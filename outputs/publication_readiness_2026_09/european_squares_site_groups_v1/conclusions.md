# Recording Isolation Keys Before Forecast Readout

2026-09-24. Result source: fresh_run for grouping; raw and official metadata
identities cached_verified against the preceding complete audit and replay.

## Implemented Isolation

All 376 raw recordings now have an explicit locality-group key bound to their
source member, square ID and complete parsed-row SHA256. The rule uses only
the official place overview, camera-reference identity and comparative-table
site locations. It does not read prediction errors, future endpoints, labels
for easy/hard classification or model choices.

Same-city/country squares, identical camera references and geographically close
sites within 5 km are joined transitively. All dates and both collection series
for a location stay together. Missing coordinates are not replaced by zero.
The season table's conflicted coordinates and IDs are not used to relocate raw
recordings. Original raw filename IDs and the official overview stay explicit.

39 nominal square IDs form 37 provisional locality groups. Trencin IDs 67/68
and Brzesko IDs 104/105 each share a group. Their official site positions are
approximately 109 m and 250 m apart, respectively. These are geographic
distances between published site metadata, not calibrated trajectory distances.
All 39 camera references differ after URL normalization, which does not prove
different camera hardware, fields of view or online stream ancestry.

A 1 km proximity rule also gives 37 groups; a 10 km rule gives 36. The 5 km
screen is a conservative working grouping rule, not an independence theorem or
a tuned safety radius. The wider-radius sensitivity must remain visible in
later generalization analysis. No role is selected using these data's outcomes.

## What This Does Not Complete

- No train/selection/calibration/confirmation roles have been allocated yet.
- Partial recording duplicates and related-source prior exposure remain to be
  checked. Full-track exact aliases were absent in the raw audit but that does
  not resolve all source reuse.
- Metadata date and recording correspondence conflicts remain recorded; grouping
  all same-place recordings prevents a simple temporal split from disguising them.
- No verified frame-to-seconds mapping, ground-plane trajectory transform or
  online sensor-as-of causal provenance is established.
- Independent confirmation, model improvement and calibrated protection remain
  unproven. No training, forecast scoring, Stage5C or SMC occurs.

## Reproduction

```bash
.venv-pytorch/bin/python -m pytest tests/test_m3w_european_squares_grouping.py -q
.venv-pytorch/bin/python scripts/group_m3w_european_squares_sites.py
```

Six grouping tests pass: transitive locality, missing-location handling,
camera-URL aliases, ordering stability, outcome independence and invalid
identities/coordinates. Tests verify the implemented screen, not geographic
independence. The generated [site_groups.json](site_groups.json) is a concrete
input to the next exposure and role audit, not a replacement for that audit.
