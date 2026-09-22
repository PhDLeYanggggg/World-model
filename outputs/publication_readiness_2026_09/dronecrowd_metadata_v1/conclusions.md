# DroneCrowd: Source Identity Before Forecast Admission

## Material Passport

2026-09-22. `fresh_run`: acquired five small official release files, totaling
6,649 bytes, and audited their identities, clip lists and conversion semantics.
`cached_verified`: offline hash/analysis replay and a separate CSV recount both
agree. `not_run`: annotation-archive acquisition, real XML inspection, physical
site assignment, feature conversion, training, calibration and forecasting.

This is a data-interface prerequisite, not an external generalization result.
The previous 4.09764% development ADE improvement and failed per-scene easy
gate are unchanged. Stage5C and SMC remain off; submission readiness is unmet.

## Evidence That Changes the Next Action

The actual official lists have 82 train and 30 test sequence IDs. Their union is
exactly IDs 1 through 112, with no duplicate entries or train/test ID overlap.
The release README, line 9, nevertheless states that val is sampled from test.
Thus clip-ID partition verification does not approve the packaged val/test as
independent selection, calibration and confirmation material. Actual val image
membership was not inspected, and distinct clip IDs do not establish distinct
physical sites. No scientific roles were assigned.

Two supplied converters explain an additional compatibility risk. `saveGT.m`
line 13 uses XML frame `i-1` for output image `i`, and line 16 increases the agent
ID by one. It saves a bounding-box center rather than a ground-plane footpoint.
`xml2mat.m` filters outside/occluded records; the derived six-column array does
not retain those flags or per-track labels. Its separate `label` variable is
overwritten for each track. It also prefers an existing MAT without checking the
XML source identity. A later importer must not silently mix these IDs, treat
filtered gaps as observed continuity, or substitute an unbound MAT cache.

These facts come from static reading of the official source, not running it.
They do not establish whether the actual XML contains keyframe flags, whether
annotations were interpolated with later information, or whether camera motion
can be separated from agent motion. An existing keyframe flag alone would not
establish online causal provenance either. Those are raw-data audit questions.

## Implemented Checks and Their Limits

- The acquisition entry point can fetch only five metadata IDs. Archives, XML,
  images and videos are outside its allowlist. HTML/sign-in/confirmation pages
  are rejected rather than treated as successful downloads.
- The first acquisition stores exact bytes under ignored `external_data`, with
  SHA256 and official URLs in a light manifest. Later local verification refuses
  changed bytes. This freezes our snapshot; Google file IDs are not immutable
  release versions or signed provenance.
- Sequence parsing rejects duplicate/aliased IDs. The reviewed derived-ID bridge
  checks the frame range and namespaces identities by recording.
- A limited release-role check rejects assigning official val and test to
  different independent roles. It is not a substitute for the full experiment
  lineage contract, and passing it grants no admission.
- A provisional local XML structural screen preserves visibility/attribute
  counts, checks IDs, frames and boxes, and refuses entity declarations. It
  neither interpolates nor exports coordinates or future labels for a model.
  **Only synthetic fixtures have exercised it. Actual DroneCrowd XML remains
  unread, so real parser compatibility is not yet verified.**

The raw annotation download remains subject to the earlier unanswered Google
cannot-virus-scan warning confirmation. That warning was not accepted; it is
not bypassed by this metadata-only fetch. No supplied MATLAB or Python code was
executed. The browser folder is retained for continuation, not as completed data
acquisition.

## Verification and Reproduction

121 scoped tests pass in 2.02 seconds: 54 new metadata/XML-fixture/recount tests
plus the existing intake and experiment-contract tests. This is not a full
legacy-suite pass or proof of model accuracy. The separate verifier does not
import the primary parser; it recounts lists with CSV, verifies byte identities
and locates the reviewed statements in the original source files. It is a
separate implementation in the same workflow, not independent team replication.

Analysis SHA256:
`1328fd4e7945c10f9bb8cb60bd48e3f1299415b9dd9617f96fd0e68bff34f6b8`.
Manifest SHA256:
`1856dee2577647342a975a1ef449a8f2b52bdbb1fc56f324a6e38170cc2334ee`.

```bash
.venv-pytorch/bin/python scripts/audit_m3w_dronecrowd_metadata.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_dronecrowd_metadata.py --verify
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_dronecrowd_intake.py tests/test_m3w_dronecrowd_verification.py tests/test_m3w_intake_admission.py tests/test_m3w_experiment_contract.py
```

First acquisition uses `--download-metadata`; it does not fetch annotations.
After separately approved acquisition, `--inspect-xml PATH --sequence-id ID`
can screen an explicit local original XML. It remains diagnostic and does not
approve roles, choose sampling stride, construct goals, train or access model
test scores. Do not substitute a guessed XML fixture for the real archive.

## Research Consequence

The next raw-data check is now specified by source evidence: original identity
and visibility semantics, interpolation/source-time provenance, physical-site
grouping and camera motion. Acquisition alone cannot supply independent scene
support or justify metric/seconds claims. The independent-source versus strict
nested-development research choice remains open; neither path was silently
authorized here. No new CREATE job was submitted or remote queue verified.

See [source hashes and URLs](source_manifest.json), [analysis](analysis.json),
[separate verification](separate_verification.json) and [release audit](release_audit.md).
Primary release: [official DroneCrowd folder](https://drive.google.com/drive/folders/1EUKLJ1WmrhWTNGt4wFLyHRfspJAt56WN).
