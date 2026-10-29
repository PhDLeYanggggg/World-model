# Experiment Data Card

## Population and Task

318,969already-opened European SOURCE windows from163recordings and12localities.
Rows are forecast windows,not unique agents. Windows overlap. Eight observed
steps and twelve requested future steps use raw stride12. Encoder time indices
are not verified seconds. Raw-frame t+50is a separate historical task.

There are311,922rows with at least one future label,193,705with complete
requested futures,and7,047with no future labels. All remain visible in the
accounting. ADE uses available valid labels; FDE is undefined without its
final-step label. Unknown outcomes are not filled with zero or filtered at
inference by label availability.

Coordinates are image-local pixels. Labels are released detector tracks,not
human-gold trajectories. Homography,metric scale and effective seconds are not
verified here. Relative image-coordinate results are not physical-safety proof.

## Roles and Exclusions

Every locality here is design-exposed source development. Each frozen producer
fits four source localities and predicts the other eight,under the inherited
three-fold/three-seed protocol. That fitting exclusion is preserved but does
not make these historically explored sources independent confirmation.
Independent selection,calibration and confirmation remain unopened.

The148-view support probe uses six fixed seed43producer/controller contexts,
24SOURCE TRAIN packets and74unique source-qualified recordings. Physical
packets also contain previously opened original-TRAIN inner-held rows; the
unchanged whole-record fitting filter is applied before diagnostics. This is
not a new held evaluation. Repeated views are not independent recordings.

## Inputs and Supervision

Inputs are past-indexed ego/neighbor positions,masks,causal motion rollouts
and request indices. Future positions and masks serve loss/evaluation only.
No future endpoint,central velocity,test-endpoint goal or test normalization
is introduced. No new goal/image annotations are constructed.

The current model forecasts each target using local masked neighbors. This
does not establish all-agent joint dynamics or a new multimodal world state.
Cached source/producer/partition lineage is hash-verified,and newly sampled
training rows exclude each producer's held sources. No fresh raw-perception
or tracker-causality audit is claimed.

## Storage and Release

Data and weights remain outside Git. A transport-only private cache stores
the exact24original TRAIN packet byte streams,verified by original size/SHA.
It uses about420MiB,with4GiBcap and10GiBdisk reserve. The first SSH interruption
and subsequent recovery are recorded; no HPC job was submitted or restarted.
Only code,aggregate diagnostics,hashes and reports are published.
