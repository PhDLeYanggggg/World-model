# First-View Inspection

On 3 November 2026 I inspected all ten private overview sheets, pages 01-10,
covering the first selected view of every one of the 120 MovingDroneCrowd++
clips. This is a model-assisted visual self-audit, not human gold or a review
of all 360 views. The contact-source manifest is
`overview_contact_sources.json`, SHA-256
`3fa5d1cc16f4942f4814bff233baf5ab8ac2a6733e5ada5bcb67acbcb95e8b9b`.
It records the exact private PNG hashes, clip IDs, actual frames and JPEG hashes.
The images are not committed or redistributed.

The overview shows oblique as well as overhead views, substantial perspective
variation, rotated camera axes, low-light recordings, water and repetitive
paving/roof structures. The portrait-axis repair removes the verified large
resize distortion; it does not make every camera upright, recover ground-plane
geometry, stabilize motion or remove parallax. The scene14 first views remain
sideways after the fixed axis convention. No orientation is hand-chosen from
matching outcomes.

Adjacent publisher clips often show visibly similar streets, plazas or beach
backgrounds. Their existing publisher grouping is retained. This inspection
neither assigns geographic sites nor establishes that different groups are
independent. The actual complete matching run is still in progress; no selected
cross-source pair panel or full source-exclusion result has yet been inspected.
Dark/sparse/repetitive backgrounds remain limitations, not evidence of no overlap.

The renderer originally saved all ten PNGs and then failed while writing its
manifest because the v2 module did not export a helper. Only the unbound
renderer was repaired. Explicit resume recomputes the pixels, requires equality
with the saved images and leaves their original bytes unchanged. Its regression
test passes (one test, 0.22 seconds). The frozen acquisition/matcher definition,
criteria and pair receipts were not modified. No predictive outcomes, source
roles, protected reservations or model deployment changed.
