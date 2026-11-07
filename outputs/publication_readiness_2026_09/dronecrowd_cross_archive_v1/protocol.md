# Label-Blind Cross-Archive Source Screen

Before any DroneCrowd predictive role is assigned, screen the external recordings
against the image roster used in the existing MovingDroneCrowd producer-A and
exposed-B experiments. The original published DroneCrowd release folders have
known cross-folder matches and are not scientific roles. This screen addresses
an additional, different dependency: possible overlap with the old project
development recordings. It does not repeat the within-DroneCrowd audit.

Use all nodes from both existing registered raster rosters. Deduplicate their
exact shared frames by recording ID, annotation frame and source JPEG hash.
For each node, use the earliest, middle and last available cached background
raster, selected before matching and without forecast errors. DroneCrowd uses
the 112 existing three-view masked RootSIFT packets, hash-verified from the
completed grouping audit. The cached Moving grayscale/mask pairs are resized
to the same coordinate raster and converted using the original CLAHE and
RootSIFT settings. No source images, future masks or annotation arrays are
exported as model inputs. This is source-audit image use only.

Run the original mutual-ratio/RANSAC matcher with all its fixed ambiguous and
strong thresholds, on each cross-archive recording pair and up to nine view
combinations. As in the parent audit, two distinct strong matches can stop a
pair early. Retain every candidate match for source review. Any ambiguity or
strong evidence quarantines the entire corresponding external exclusion group
for the proposed transfer experiment; do not remove just the matched clip.
The checked 45 conservative within-archive groups remain intact.

**Scope:** negative image matching does not certify independent physical sites
or exclude nonoverlapping views of the same site. Different raster resolutions
and partial frame sampling can reduce sensitivity. The comparison covers the
specified A/B image rosters, not every historical dataset ever used by M3W.
No scientific role assignment, calibration, training, inference outcome,
support-rule or 2% budget change follows automatically from this screen.

Use local arm64/OpenCV with one matching thread, no new large downloads and
atomic per-recording receipts. The first full recording is a timing pilot
included in the final screen. Separate verification rechecks the input hashes
and aggregates all saved receipts, not a second fresh feature-matching run.
Raw images, per-pair descriptor caches and credentials stay out of Git.
Stage5C execution and SMC remain off.
