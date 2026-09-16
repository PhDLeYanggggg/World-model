# M3W Asset Inventory and Local Training Verification

Date: 2026-09-16. Scope: source/data engineering and runtime verification, not a new forecasting result. Research target: CVPR 2027, not AAMAS. No submission-readiness claim.

## Existing Assets

| Location | Observed asset | Evidence and limitation |
| --- | --- | --- |
| Local | `external_data/OpenTraj`, approximately 1.3 GiB | Available. Nine canonical recording files rebuilt; data aliases are grouped, not counted as independent datasets. |
| Local | `data/stage24_sdd_fast_cache`, approximately 867 MiB; `data/stage26_sdd_feature_store`, approximately 11 MiB | Directories available. No new SDD accuracy or full SDD lineage claim in this round. |
| Local | `outputs/stage37_t50_history`, approximately 144 KiB | Historical reports available; independent external deployment evidence is under revalidation. |
| Local | Stage41/42/43 reports and weights | Approximately 20/39/17 MiB in the inspected output directories. Existing evidence retained, not newly reproduced. |
| Local | `outputs/stage44_worldcore`, approximately 3.4 MiB | Seven checkpoint identities and twelve input-cache hashes were verified by the earlier same-day audit. Hash agreement does not remove recording leakage or teacher exposure. |
| GitHub | `PhDLeYanggggg/World-model`, `main` | Read-only remote query returned `24a145c2f889cef3064a87e5b1b6bff7a6faea9d`, matching local committed HEAD before this patch. This is a point-in-time check, not a check of future commits. |
| CREATE | SSH endpoint reachable, M3W asset contents unknown | Authentication ended with `Permission denied (publickey)` and a portal MFA notice. Directory/scheduler/resource inspection not_run; no new jobs submitted. Project path and connection repair requested. |
| Local disk | Approximately 102 GiB free | Current snapshot only; not permission to materialize large datasets. |

There are 3,019 pre-existing unrelated staged data-lake/cloud changes. This work does not include or reset them. Large rebuilt data and runtime checkpoints remain in the ignored `data/stage_cvpr2027_causal/` tree.

### Other Local Sources: Raw Audit Added

The earlier filename-only inventory below is superseded in scope by the [fresh raw-source audit](external_source_audit/source_review.md). GC / HERMES / Wild-Track / CITR / VRU now have read-only parser results: 24,745 usable tracks, 3,807,482 points, no inferred metric/time conversion, no new official split. GC has no exact raw t50; VRU has no verified global frame identity and two invalid-clock tracks. CITR's local pedestrian count differs from the author description, but all 344 raw CSV blob hashes match the author's current commit: no local files are missing relative to that version. Controlled trials and camera views remain distinct from independent physical scenes. Source terms/exposure are not yet closed; there is no new untouched confirmation source. This is actual data inspection, not merely another filename list and not a forecasting experiment.

### Earlier Filename-Level Snapshot

A filename-level scan also found the following local material outside the nine-recording rebuild. These are **file counts, not verified rows, independent scenes or eligible experiments**. No new download, conversion, training or validation on these sources was run.

| OpenTraj directory | Local files observed | Current status |
| --- | --- | --- |
| GC | 12,686 `.txt` including calibration text, plus one calibration JSON | Annotation files present; inspect previous exposure, identities, labels and licensing before selecting a holdout. |
| HERMES | 52 `.txt` including calibration text | Local README describes controlled pedestrian experiments, not unconstrained natural scenes. Do not equate laboratory success with natural-scene generalization. |
| Wild-Track | 400 annotation JSONs, calibration XML files | Annotation/calibration material present. Local README says license information is unavailable; current rights and usable trajectory identity need verification. |
| CITR | 408 CSV / 76 TXT | Presence only; source, geometry, agent classes, repeated trials and prior project usage not audited here. |
| VRU | 1,562 CSV | Presence only; schema and recording independence not audited here. |
| ATC / InD / L-CAS / Edinburgh | Documentation, reference material or download instructions in inspected paths | No eligible local trajectory source established by this scan; directory existence is not dataset availability. |

These findings mean that "no external data exists" would be too broad. The unresolved question is which sources can supply legally usable, appropriately independent confirmation. Local documentation is not itself a verified metric/time calibration. Dataset acquisition and final holdout choice still require the stated evidence and protocol checks.

## Fresh Causal Rebuild

The new reader reconstructs inputs from raw positions, not legacy feature/teacher outputs. Source aliases are catalogued by recording; nine enabled recordings belong to six physical-scene groups. Incomplete challenge excerpts are quarantined. Numerical equivalence of named cross-format aliases is not assumed.

- Raw points: **76,619**.
- 8-observation / 12-prediction-step windows: **27,053**. This is a candidate view, not an approved official protocol.
- Exact raw-frame windows: t10 **48,286**; t25 **0**; t50 **38,733**; t100 **28,330**.
- Total indexed views: **142,402**. Views overlap and are not independent statistical samples.
- All window boundaries, continuity, source positions and exact horizons checked.
- **144 real-window counterfactual checks**: replacing every agent's post-current-frame positions with NaN left every inference input and baseline rollout unchanged.
- **36 real-scene counterfactual checks** also preserved agent membership and every input. The new scene query includes all currently visible agents with sufficient continuous past support, even when future labels are missing. Labels and their availability masks are accessed separately for loss/evaluation. This provides the scene-level interface; it does not establish a trained joint policy.
- No teacher, goal construction, official split, future-derived normalization or future endpoint input in this rebuild.

This establishes a usable reader, not a complete experiment no-leakage certificate. Split assignment, within-split teacher refitting and independent confirmation remain pending. A longer `UCY/students01/students001.txt` source is present (21,813 points / 415 agents, versus the canonical cropped representation). The existing `outputs/stage42_long_research/ucy_students_t50_source_support_stage42.json` already records this source; it is not a newly discovered or untouched domain. Its representation/provenance needs review before replacement, and earlier horizon counts must not substitute for the new exact-frame checks.

## Fresh Runtime Results

Actual training: 68,292-parameter two-layer Transformer, 1,024 real past-history windows, batch 128, 4,000 optimizer steps per backend. The objective reconstructs masked **historical** positions. No future labels are read. This is an engineering workload, not world-dynamics evidence; its weights must not be reused as confirmatory pretrained models.

| Backend | Compute threads / workers | Training seconds | Rows/s | Peak process RSS MiB | Resume next-step parameter difference |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU | 4 / 0 | 29.81 | 17,178 | 391.44 | 0 |
| CPU | 8 / 0 | 36.03 | 14,210 | 370.38 | 0 |
| MPS | 4 / 0 | 75.55 | 6,777 | 379.58 | 0 |

All losses/gradient norms remained finite. CPU4 and MPS also resumed in a new process from step240. Each backend saved optimizer/model state and reproduced the next parameter update after reload. Interop threads=1; heartbeats and atomic checkpoints present. Torch 2.12.0 / arm64 uses Accelerate BLAS and USE_MKL=OFF.

After adding the scene query API, all three runtime input signatures were regenerated and still matched `32425db4160f0b141f6a13927a8badd63dc16a40233628fbfa9d5e9c5356d849`; the existing runtime measurements remain tied to identical arrays.

Timing includes checkpoint IO and scalar synchronization; MPS includes warmup and two processes, and RSS is not total GPU/unified-memory allocation. These single measurements are not a general hardware ranking. CPU4 is the provisional choice for this small workload. Full-scale throughput, hours-long stability, larger-batch MPS behavior and CREATE performance are not established.

The first sandboxed MPS allocation failed with a macOS-version error despite macOS15.3.1. A sandbox-external authorized retry completed real MPS training. This failure is recorded, not silently converted into a CPU success.

## Priority Gaps

1. **Scientific protocol decision:** approve primary observation/prediction lengths, recording/physical-scene folds, selection versus calibration independence, primary metric and risk budget. Standard-protocol main tables and raw-frame supplement are proposed, not assumed approved.
2. **Clean supervision:** refit baseline selection/teachers only within training folds; use actual realized neural-minus-baseline errors. Old test-exposed weights cannot become independent evidence by renaming the split.
3. **Minimal method experiment:** fixed candidate forecaster; compare no gate, independent-agent, scene-uniform and joint intervention at matched coverage/risk. Then scale only if the mechanism merits it.
4. **Scene-level statistical support:** six current physical groups are a limitation for calibration, held-out generalization and uncertainty. Large overlapping window counts do not solve this. Formal main results still require three seeds and cluster-aware reporting.
5. **CREATE assets/access:** inspect existing M3W runs and resource constraints after authentication; avoid duplicate jobs. Meanwhile local development is operational.
6. **Paper evidence:** matched public baselines, independent confirmation, risk/consistency ablations, figures and the full reproducible paper remain incomplete.

## Resume Point

Latest continuation: the [causal supervised backend](supervised_backend/implementation_and_limits.md) now connects the verified recording reader to a fixed-budget Torch forecaster and out-of-fold realized benefit/harm learning. It does not use old teacher outputs. Atomic checkpoints include optimizer, sampler and RNG; OOF fold caches are verified on resume. CPU and MPS synthetic recovery passed, and 24 real-input queries / 345 agents remained unchanged under future-position corruption with zero label calls. Seventeen new tests bring the current combined focused set to **102 passed in 11.65 s**. No real forecast training, development selection, calibration or confirmation has run through this backend. The MLP cost interface is implemented, while the trained synthetic cost control is ridge. The next missing experiment connection is development-only selection and matched-budget evaluation, not another source inventory.

The [experiment contract](experiment_contract/implementation_and_limits.md) binds scientific choices, recording roles and recursive fitted-artifact provenance. It freezes a calibration family and a confirmation run before their respective reader access, with identity-preserving resume. The current real-asset draft remains unapproved/unassigned and correctly refuses both new training entry points with exit 2. All nine current recordings are historically exposed development material; no untouched holdout was created. This interface is not yet wired into every legacy trainer and cannot prove truthful declarations or IID.

Verification: previous full suite 1,870 passed / 1 failed in 73 minutes; the earlier focused joint-intervention/reader set was 48 passed. The independently reproduced full-suite failure is in the pre-existing data-lake fixture's missing manual handoff. Legacy test-written reports/state were quarantined and recovered; no replay was promoted to new research evidence. See [verification and isolation record](verification_record.md). The full suite remains non-hermetic and was not rerun for this scoped change.

Completed: raw-position reader/index, source identity checks, exact horizons, counterfactual input checks, CPU/MPS actual training and optimizer-state recovery, Chinese engineering runbook. The subsequent [joint-intervention prototype](joint_intervention/method_and_checks.md) now includes five control arms, common-coordinate restoration, realized relative-cost labels and a bounded cluster-risk screening primitive. Eighty synthetic solver problems match exhaustive solutions; 36 real scene queries and 497 agent queries pass coordinate checks. It is not a fitted or risk-certified policy. No official forecasting experiment launched. No new deployable model, no metric/seconds claim, Stage5C off, SMC off.

Next: resolve the pending protocol choice and CREATE project path, verify eligible synchronized source release/terms/exposure, then fit a clean fixed predictor and split-bound gain/harm heads for the implemented control-arm comparison. The preceding raw-source audit had 85 selected regression checks; the new combined set above supersedes that count without implying a full-suite pass. Joint energies alone are not novel; see the [focused prior-work review](joint_intervention/related_work_constraints.md). Do not train from rejected legacy caches. See [runbook](local_create_runbook_zh.md), [latest causal checks](joint_intervention/causal_recording_checks.md), and the three `runtime_*.json` files for executable evidence. The source audit did not change the existing runtime or contract-bound modules.

Independent source parsing is now complete for this local five-source snapshot, and CITR's raw snapshot has been verified against the author repository. Remaining eligibility work is narrower: documentary count discrepancy, original use conditions, prior exposure and global scene/time identity where missing. Do not inspect model performance to pick an apparently favorable confirmation domain. Presence of files or a clean metadata gate is insufficient to designate an independent benchmark.
