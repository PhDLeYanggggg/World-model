# Execution and Verification Record

## Experiment Passport

- Experiment: `zero_atom_v1`, 2026-09-24.
- Analysis SHA256: `9fd090402ac1f0e9334aab4c86d5c3439af6d89e77d081de458a7913b9a784d4`.
- Full registered matrix: 36 leaf readouts on 36 cached-verified forests,
  128 trees each; 175756 past-eligible windows and 188388 query/action/seed decisions.
- Scientific outcome: negative. The extra component veto reduces utility and
  fails zero-reference protection. No deployment or independent-confirmation claim.
- Protocol: obs8/pred12, stride12 annotation steps, annotation pixels, four
  development-exposed sites and three seeds. Not historical raw t50.

## Actual Local Runs

| Step | PID | State | Logged elapsed seconds |
|---|---:|---|---:|
| Real 16-tree pilot | 3434 | Saved checkpoint, exit 0 | 50.1334 |
| Resume all readouts, choices and aggregate | 3535 | Completed, exit 0 | 729.8317 |
| Full decision replay | 4720 | Exact arrays and records, exit 0 | 435.5113 |
| Full aggregate replay | 5258 | Exact aggregate hash, exit 0 | 75.2948 |

The readout fit plus in-source prediction/support loops total 225.4343 seconds,
including saved pilot fitting. Main phase time is 12.16 minutes, excluding pilot
setup and subsequent verification. This was the full fixed experiment, not a
quick sample or scale downgrade. There was no new neural forecast training and
no new forest partition fitting. All original targets, draws, source exclusions,
cutoffs and risk allowances remain bound by hashes.

Native arm64 `.venv-pytorch`, CPU4/interOp1/workers0, single-process loading,
checkpoint every 16 tree readouts, exact-identity resume and a single writer lock.
A sampled process used about 7.7 GiB RSS during decision solving; this is not a
  measured peak. Available disk was about 42 GiB. No resource probing or
multiprocess DataLoader was introduced. Slow healthy work was allowed to finish.

The last retained bounded CREATE access attempt failed authentication. Current
remote jobs/assets are unknown. Local resources suffice for this experiment;
no remote job or simulation-project artifact was submitted, cancelled or changed.

## Verified Checks

- 53 scoped tests pass in 1.83 seconds. These include all 13 new module tests;
  the earlier 13-test invocation overlaps and is not counted again. No unchanged
  full legacy suite was rerun.
- New tests cover event labels, unknown-label exclusion, exact source weights,
  frozen leaf support, incomplete readouts, strict zero-probability admission,
  count/risk/objective checks and matched feasible-incumbent handling.
- Every leaf's weighted support matches its frozen forest exactly. Unknown
  label draws are zero. Source-excluded fits retain the complete producer chain.
- Separate arithmetic verifies 36 fits, 565164 matched query-count comparisons,
  753552 original-unit risk checks and 360 fresh scene reductions. All 12 reused
  aggregate rows reproduce exactly. It is a second implementation run by the
  same executor, not independent scientific review.
- The 753552 solver records contain two uncertified-optimal statuses, zero risk
  violations, zero count mismatches and zero retained matched incumbents.
  Exhaustive optimality is not run. Prior experiments' failed count controls
  remain failed; this experiment does not retrospectively repair them.
- Complete decision-array/record replay and aggregate replay are exact. All
  listed PIDs are absent after successful exit. All required processes finished.
- Reporter and separate post-readout diagnostics execute successfully. The
  aggregate SVG was rendered and visually inspected: labels, bars, harmful-case
  counts and 2% ceiling are visible without clipping. Its preview PNG stays local.
- Post-readout diagnostics do not alter models, thresholds or selection. They
  establish rare-event misses, harmed-case probabilities, unique/repeated harm
  counts and complete moving/effective Brier. Unknown futures remain explicit.

## Preservation and Sharing

Source-only fitting and new readout are `fresh_run`; forest partitions,
forecasts, source arrays and previous controls are `cached_verified`.
External evaluation, independent calibration and confirmation are `not_run`.
DroneCrowd stays closed and IMPTC remains quarantined. No unsupported
metric/seconds/true3D/foundation/submission-readiness claim is made.
Stage5C and SMC remain disabled.

Only code, configuration, aggregate reports and an aggregate SVG are shared.
Readout weights, input arrays, future labels, per-query caches and preview images
are excluded from Git. The unrelated staged changes preserve fingerprint
`c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323`.
See the [operation guide](operation_zh.md) for resume and reproduction commands.
