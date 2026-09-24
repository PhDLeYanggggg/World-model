# Execution and Evidence Boundaries

2026-09-24. All required local processes in this work item completed normally.
No long-running training job was submitted, interrupted or relabelled complete.

| Operation | Actual execution | Evidence source |
|---|---|---|
| Full eight-frame duplicate screen | PID14846,320.92s,exit0 | fresh_run; first pilot fingerprint cached_verified |
| Signature matching replay | PID15406,14.32s,exit0 | cached_verified signature bytes; matching freshly recomputed |
| Raw review of240 quantized candidates | 27recordings,625instances,exit0 | fresh_run raw reparse |
| Pre-intake exposure search | 3,083tracked paths,91identity receipts,435aliases | fresh_run bounded search, not universal absence |
| Locality role assignment | constructed and rerun with identical hash | fresh_run decision; dependencies cached_verified |
| Source cohort pilot | PID15738,3.77s,exit0 | fresh_run one real training recording |
| Complete registered source cohort | PID15800,113.77s,exit0 | pilot cached_verified plus162fresh recordings |
| Complete source raw replay | PID15975,115.40s,exit0 | fresh_run all163recordings,all arrays exact |
| Scoped regression suite | 79passed,0.61s | real test execution, not model efficacy |
| New neural training or prediction errors | not_run | source cohort now ready; a fixed source-only experiment is next |
| Reserved outcome readout | not_run | model-selection/calibration/confirmation remain closed |
| CREATE M3W assets | not_run | simulation task confirms exact M3W directory unknown |

The archive and per-record row hashes bind every artifact. Role access tests,
actual per-record prefix checks, immutable dependencies and raw replay are
separate from scientific independence and predictive superiority. No source
screen is reported as model lift. A protected task not inspected is unknown,
not absent or failed. No remote filesystem scan or simulation modification.

Private caches hold logs, PID heartbeats, per-record receipts and NPY arrays.
The published outputs contain aggregate counts, source identities and code only.
The pre-existing3,019staged Git entries and unrelated unstaged changes are not
part of this work item. The staged-diff identity before integration is
`c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323`.
Only an explicit file allowlist may be committed. No raw/cache/media/weight/
third-party implementation or virtual environment is included.
