# Data Card: Bridge Attribution

## Verified Reuse

The source bank contains 318,969 indexed rows in 12 European Squares localities.
Three four-locality rosters contain 116,823, 23,762 and 178,384 rows. Producer A
and cost-controller B are disjoint. A produces both forecast endpoints and the
learned motion floors used as B's training teachers. Full-pair teacher chains
are inherited with hashes from the preceding bridge study. No B-trained add-only
controller supplies B's new labels.

Readout reuses six already-opened model-selection localities: 087, 092, 093,
103, 104 and 125. They contain 28 recordings, 38,102 indexed targets and 7,087
past queries. There are 21,434 complete futures, 15,820 partial futures and 848
unknown futures. ADE supports 37,254 rows; final endpoint error supports 27,694.
Unknown futures remain in inference and matched intervention-count decisions.
Only loss/evaluation consults future labels. These six localities cannot be
renamed independent confirmation data.

## Features and Units

Eight observed and twelve requested native annotation steps, stride12 in this
adapter. Coordinates are image pixels from released detector-derived tracks,
not manually adjudicated human-gold trajectories. No verified metric scale,
homography or effective-time conversion is claimed. This is not true3D,
foundation-model validation or a physical-safety study.

The 383 causal columns contain inherited past geometry, rollout-relative
diagnostics, a causal CV rollout and four frozen policy bits. Full pairs retain
the old neural policy bits. Motion-only pairs zero those two bits and remove
neural trajectory endpoints. Their learned floor scorers remain, so this is
not a neural-network-free baseline. Learned normalization, CV cost scaling,
easy/hard cutoffs and training draws come only from authorized source roles.

## Closed Roles

Twelve calibration and six confirmation localities remain unopened. DroneCrowd
confirmation is unchanged. No test endpoint goals, future target input,
central velocity, simulation-success substitution, Stage5C or SMC.
Windows and repeated producer/controller views are not independent samples;
paired bootstrap resamples physical localities after averaging three seeds.
