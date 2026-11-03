# Source-Image Inspection Scope

On 3 November 2026 I inspected all three overview pages, covering every one
of the 28 DUT author backgrounds, plus all eight generated candidate panels.
This is model-assisted self-audit, not human gold or a geographic certificate.

The overview manifest SHA-256 is
`b729d0b15db42337a8654e3a5a156057070b4eeabf3b06370ca0bc76536aecd6`.
The candidate manifest SHA-256 is
`b735341096c34f60d741ec740540d89da1e74593fb4002a3a1ad35090d45ee52`.
They bind exact private PNG hashes and recording/view identities. No images
are committed. The three-page overview has blank unused tiles after the
28th image; they are not missing source images.

The overview is visually consistent with the two author-described settings:
the crosswalk/intersection and curved shared-space pavement. Repeated road
markings, planted areas and parked vehicles are visible, with crop/scale changes
and composite artifacts. The eight selected pair panels are the first lexical
within-DUT candidates: intersection_01 against intersection_02 through _09.
Their repeated structures are consistent with retaining the existing same-site
constraint. They do not establish new physical independence, temporal identity
or duplicate pedestrian trajectories. Quarantined _04 remains excluded from
prediction, even though its source background is included in this audit.

There are no supported Moving-to-DUT or DroneCrowd-to-DUT candidate panels.
An empty cross-source candidate set is not a human-confirmed absence of overlap.
All 171 within-DUT supported recording links remain conservative ambiguous
links; no single background was replicated into artificial temporal views.

Some cropped source images are near-square; the registered 960x540 preparation
has a maximum axis-scale ratio of 1.7423. That distortion, cropping, background
estimation and viewpoint change limit negative matching. It is explicitly
reported, not silently treated as affine-invariant matching or verified recall.
No matching threshold or role was tuned from predictive errors. No composite
is allowed as a past-only model image input: its aggregation time is unknown.
