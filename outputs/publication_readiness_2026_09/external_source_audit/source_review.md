# External Source Eligibility Review

Date: 2026-09-16. Research route: CVPR 2027; AAMAS is not scheduled.

## What Was Actually Done

`fresh_run`: read the local raw annotations of GC, HERMES, Wild-Track, CITR and VRU; hash the exact input files; parse native identities, coordinates and timestamps; count continuous observation histories and exact available labels. The implementation neither calls OpenTraj loaders nor writes to the raw tree. No model predictions, learned features, new data acquisition, official split assignment or training took place.

The result is [availability.json](availability.json) and [availability.md](availability.md). Those hashes bind the local snapshot. A subsequent [CITR upstream check](citr_upstream_identity_system_tls.json) additionally matches all 344 local raw files to the author's Git tree; the other four sources have no such upstream byte-identity check yet. Neither check grants permission for all uses. This is metadata review, not confirmation-set evaluation. No inference feature is allowed to use the full-track availability examined here.

The five sources contain **24,745 successfully parsed tracks and 3,807,482 unique agent-frame/measurement points**. They do **not** provide 24,745 independent scenes. Two additional VRU tracks are quarantined for invalid clocks. The large HERMES count is mainly overlapping views of controlled trials. Formal forecast results remain `not_run_protocol_and_source_eligibility_pending`.

## Source Findings

### GC: Long Tracks, One Location, No Exact Raw t50

The 12,684 local files contain 456,410 points, with 6,824 noncontiguous edges. Consecutive annotations use increments of 20; gaps are larger multiples of 20. Strict obs8/pred12 views number 183,436, but exact raw t10/t25/t50 availability is zero. Exact raw t100 has 267,468 K=8 views. Missing raw horizons must not be manufactured by interpolation and then reported as observed labels.

The local release description matches the scale of the manually annotated walking-route corpus in [Yi et al., CVPR 2015](https://openaccess.thecvf.com/content_cvpr_2015/papers/Yi_Understanding_Pedestrian_Behaviors_2015_CVPR_paper.pdf). The linked [older Grand Central page](https://www.ee.cuhk.edu.hk/~xgwang/grandcentral.html) instead describes KLT tracklets and different resolution/frame counts. This is a release-identity mismatch, not proof that either dataset is unusable. Do not copy timing or release conditions across them without verification.

The local `loader_gcs.py` upsamples to a 10-index grid and applies a hardcoded homography plus a 0.8 multiplier. Those operations are deliberately absent from the new audit. Native stored position axes are preserved and labelled unverified. The station is one physical site; temporal subdivisions do not become new locations. Exact release identity, research-use conditions and previous project exposure still need verification.

### HERMES: Useful Controlled Interaction Stress, Not Natural Generalization

51 actual corridor experiment files, excluding `H.txt`, yield 9,844 trial-scoped tracks and 2,757,226 points. Every within-track raw frame increment is 1. There are 2,570,190 obs8/pred12 views and 2,196,118 exact raw t50 K=8 views. Median track length is 216 points.

The local README identifies laboratory pedestrian-flow experiments. Trials may share participants and physical facilities; neither 51 experiments nor two corridor configuration folders establish 51 independent natural scenes. They may support controlled interaction stress tests or curriculum after eligibility review, but cannot substitute for natural-scene external evidence.

The referenced [Juelich pedestrian archive](https://ped.fz-juelich.de/database/) did not yield a usable exact-record terms page in this check. Source descriptions report centimeter-like coordinates and a frame cadence, but per-file calibration and release correspondence remain unverified. No meter/second conversion or claim was made. The OpenTraj toolkit license is not assumed to license all third-party datasets.

### Wild-Track: Synchronized Multi-Agent Context, One Scene

400 annotation files yield 313 identities and 9,518 points. Nominal filename/frame increment is 5, with 34 within-track gaps. There are 4,558 obs8/pred12 views and 4,926 exact raw t50 K=8 views. The new parser decodes the native grid using integer row division, `positionID // 480`. The bundled loader uses floating division for this row coordinate; importing it would change geometry.

The [original EPFL description](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/) distinguishes original videos, extracted frames and the annotated subset. Its seven cameras are synchronized views of the same location, not seven independent domains. The local filename-to-original-video/time mapping and calibration were not validated by this metadata audit; native grid coordinates remain non-metric here.

Previous project code already parsed this source diagnostically: `src/stage42_local_calibrated_source_support_intake.py`. The subsequent guarded conversion preflight and Stage43 terms packet left conversion pending source conditions. This is not an untouched discovery. No new conversion or source-use approval has been inferred from the website's download links.

### CITR: Technically Promising Joint-Interaction Stress Data, Controlled Clips

38 local raw clip directories contain **318 pedestrian and 26 vehicle tracks**, 95,648 points. Raw frames are contiguous within each track. Strict obs8/pred12 views number 89,112, and exact raw t50 K=8 views number 76,040. Pedestrian and vehicle identity namespaces are kept separate; `p1` and `v1` are not one agent. The 64 filtered CSV files are not counted as additional recordings, and no supplied filtered velocity is used.

The [author repository](https://github.com/dongfang-steven-yang/vci-dataset-citr) describes 38 controlled parking-lot clips, six interaction scenarios, and 340 pedestrian trajectories. It distinguishes raw object files from filtered data and describes coordinate conversion. A fresh Git tree comparison at commit `b6a980628f48e7c80c748c3e333979ca3a6b0301` found exactly 344 raw CSVs (318 pedestrian / 26 vehicle), with **zero missing, extra or byte-mismatched local files**. Thus the local raw snapshot is complete relative to that author commit; the advertised 340 is a documentation/representation discrepancy, not evidence of a missing local download. Its explanation is not established. The same location and directed participant routes make a clip holdout a weaker test than independent-site generalization.

CITR is a useful technical candidate for the joint-intervention mechanism because synchronized positions and vehicle/pedestrian interaction are present. This priority is based on modality support, not prediction performance. No source has been selected as a final test set. Raw version identity is now verified against the author commit; conditions, prior exposure and scale correspondence remain unresolved.

### VRU: Single-Agent History, Not Yet a Synchronized Scene Dataset

After excluding `.svn/format.csv`, there are 1,562 actual trajectory files: 1,068 pedestrians and 494 cyclists. Two cyclist files (`waiting/108.csv`, 81 rows; `waiting/305.csv`, 92 rows) have nonincreasing timestamps and are excluded from usable counts. No silent replacement of their clocks was made. The remaining 1,560 tracks have 488,680 points; 1,558 of these start at timestamp zero. Per-track modal native timestamp increments are 0.02 for pedestrians and 0.08 for cyclists. There are 430 deviations from joint measurement/time continuity.

The [university source description](https://www.th-ab.de/hochschule/organisation/organisationseinheiten/labor-fuer-kooperative-automatisierte-verkehrssysteme/trajectory-dataset/) describes one-object CSV files, 1,068 pedestrians and 464 cyclists. Its advertised total is 1,532, thirty fewer than the local actual files. No byte-identical or exact-geometry duplicates explain the difference in this local audit. The newer extended release must not be assumed identical to these files.

Measurement IDs and local timestamps are not a verified shared video clock. Consequently raw t10/t25/t50/t100 counts are **not_run**, not zero and not relabelled observation-step counts. Per-track obs8/pred12 availability is 454,032, but artificial synchronized neighbors must not be created by matching each object's time zero. Recover original global recording/time identities or limit the source to explicitly single-agent work.

### DUT: Reference Material Is Not Trajectory Availability

Only a reference image was found in the inspected local DUT directory. The [author repository](https://github.com/dongfang-steven-yang/vci-dataset-dut) describes a separate natural campus collection; it was not downloaded or audited here. It remains an acquisition/eligibility option, not an available benchmark or a completed experiment.

Later 2026-09-16 update: the separate [DUT raw intake](../dut_causal_intake/implementation_and_limits.md) acquired and verified the author's pinned unfiltered annotations and converted them for diagnostic access. It found 457,686 points, two physical-site groups and a duplicated pair of agent trajectories requiring quality quarantine. The earlier metadata-only snapshot is preserved here; it is no longer the latest local availability status. Formal source conditions, roles and independent-test eligibility remain unapproved, and no prediction result has been produced.

## Exposure and Eligibility Boundaries

The bounded source search covered `src/`, `configs/`, the current recording catalog and relevant Stage42/43 source-intake files. GC/VRU also occur in general dataset discovery/catalog context. This does not exhaust historical caches, remote CREATE use or every experiment log. For all five sources, **prior predictive use remains unknown**, except that Wild-Track diagnostic parsing is explicitly known. Absence of a filename match is not proof of an untouched test source.

The audit found no byte-identical source files or full-track geometry duplicates within each selected raw source. It did not prove absence of cross-format aliases, recurring participants or cross-source overlap. No independent-scene count has been certified. Formal calibration and confirmation are therefore not unblocked by the raw row counts alone.

No research-use permission is granted by this report. Unresolved source conditions are recorded as unresolved, not as a legal conclusion that every possible local analysis is prohibited. Original release terms and the intended use should be reviewed before promotion into a formal training/evaluation protocol.

## Next Actions Driven by These Findings

1. Resolve the main temporal protocol and independent scene roles already awaiting the user's decision. A universal raw t50 protocol excludes exact GC labels and is undefined for VRU; an observation-step protocol is also not a common physical-time protocol without a cadence audit.
2. Follow up CITR's documented-versus-published pedestrian count and source conditions, and Wild-Track's exact release and previous use; these are the clearest local synchronized inputs for interaction-method development. CITR no longer needs a missing-download repair. Do not give either source confirmation roles until provenance and user-approved grouping are settled.
3. Seek genuinely different capture locations for independent calibration/confirmation. Controlled trials, seven views of one site, and overlapping trajectories cannot replace independent scenes. Investigate DUT or other original sources on data identity/conditions, not on model scores.
4. Preserve the two VRU failures and GC missing horizons. Do not fix them by inventing timestamps, interpolation labels, or a new physical unit claim. Use HERMES only under an explicitly controlled-data role if admitted later.

No new predictor was trained. No legacy accuracy was reproduced or repaired by this source audit. No deployment upgrade, true-3D/foundation claim, Stage5C execution or SMC.

## Verification

Fifteen new parser/window-counting tests plus related contract, joint-intervention, causal-reader, lineage and WorldCore regression checks: **85 passed in 2.69 s**. Tests cover gaps, exact horizons, duplicate/conflicting agent-frame identities, native grid decoding, class ID separation, clock failures, hidden `.svn` metadata exclusion, source immutability, and explicit timestamp-support enumeration. An independent simple line/triplet/CSV/JSON recount matched GC 456,410, HERMES 2,757,226, Wild-Track 9,518, CITR 95,648 and VRU 488,853 total rows; subtracting the two rejected VRU tracks (81 + 92) gives 488,680 usable rows. This recount is a parser check, not a model evaluation or independent-scene certificate.

The unchanged non-hermetic full suite was not rerun. Its previously recorded 1,870 pass / 1 unrelated data-lake fixture failure remains visible. The new audit does not change the three modules bound by the existing unapproved experiment contract.

The first upstream request failed because this Python environment could not validate the local CA chain; [the failure is retained](citr_upstream_identity.json). Using system curl's normal certificate verification succeeded. TLS verification was not disabled, no token was requested, and only public commit/tree metadata was downloaded. The successful report binds the script hash and author commit; it is a source-identity check, not a predictive result.
