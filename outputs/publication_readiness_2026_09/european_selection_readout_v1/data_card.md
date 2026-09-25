# Six-Locality Selection Cohort

The preassigned model-selection role comprises localities087/092/093/103/104/125:
28 recordings,6,179,261 raw rows and23,584 recorded tracks in the role manifest.
These are locality groups formed by the earlier conservative location/recording
audit, not a mathematical proof of population independence or universal prior
nonexposure. Label provenance is released detector tracking, not human gold.

The conversion uses unsmoothed bounding-box centers. Observation offsets are
-84,-72,-60,-48,-36,-24,-12,0; future labels are requested at12 through144 in
increments of12. These are raw annotation-frame offsets, not verified seconds.
Image pixels are not metric coordinates. The existing SDD main protocol is unchanged.

Up to256 uniformly spaced past-supported query frames per recording produce7,087
queries and65,094 visible-agent/query rows.38,102 targets have complete eight-step
history.21,434 have complete future labels,15,820 partial labels and848 none.
Unknown labels never suppress inference. ADE uses37,254 rows with any valid future;
FDE uses27,694 rows with a valid requested final endpoint. Partial ADE is explicitly
different from a complete20-position-window-only benchmark.

All visible context is retained in the raw input cache. The unchanged476-column
encoder packs at most eight nearest complete-history neighbors; incomplete-history
agents are not silently claimed as used encoder context. No scene image, semantic
goal map or human annotation was introduced by this readout.

Archive SHA and each decoded member-row SHA match the earlier manifest.84 prefix
checks remove all rows after the query and rebuild all input fields exactly.
Inputs and future labels live in separate arrays, and inference loads input keys
only. No future endpoint goal construction, central velocity or held-data statistics.

After this readout these six localities are development-exposed. They must never
be renamed confirmation. The12 calibration and6 confirmation localities remain
closed; DroneCrowd confirmation is unchanged. Read access is restricted by purpose,
while the original training-only accessor still rejects every reserved role.

Raw archive, per-record caches, packed inputs/labels and predictions remain private
and ignored by Git. Only code, config, hashes and aggregate evidence are published.
