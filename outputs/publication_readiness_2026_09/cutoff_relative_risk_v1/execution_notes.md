# Completed Local Execution and Verification

## Material Passport

- Experiment: `cutoff_relative_risk_v1`, 2026-09-24.
- Status: full registered fit matrix and readout completed; replay verified.
- Scientific outcome: partial easy-risk representation repair, no deployment promotion.
- Scope: four exposed source sites, not independent confirmation.
- Analysis SHA256: `c6247c4c643aefcbc36072d30b35e3c72f675af047418e08290a690164638cfc`.

## Actual Runs

| Step | Process | Final state | Logged elapsed seconds |
|---|---:|---|---:|
| Real 16-tree pilot | 99530 | Checkpoint saved, exit 0 | 45.2270 |
| Resume through all training, decisions and readout | 99628 | Full phase complete, exit 0 | 1249.5884 |
| Full decision replay | 1747 | Exact arrays and records, exit 0 | 347.0549 |
| Full aggregate replay | 2181 | Exact report hash, exit 0 | 62.8952 |

The 36 fresh fits contain 4608 trees in total. Each receives the same frozen
768000 source draws as its controls; these are weighted training draws, not
millions of independent examples. Unknown-label draws remain zero. The total
fitting-loop time is 828.5308 seconds (13.81 minutes), including saved pilot fit
time. Main process time is 20.83 minutes and includes preparation/decisions/readout
but excludes pilot setup and later replays. All listed PIDs are absent after exit.

This was not downgraded for speed. The full registered 175756-window population,
four sites, three seeds and three action families were retained. Decisions cover
188388 query/action/seed instances before aggregate readout. The old 72 control
fits are `cached_verified`, not new training. No new Transformer/EqMotion forecast
model, external outcome or deployment policy was trained/selected here.

## Runtime and Resources

Native arm64 `.venv-pytorch`: NumPy 2.4.6, PyTorch 2.12.0, scikit-learn 1.8.0.
CPU compute threads 4, interop 1, DataLoader workers 0. Forest fitting uses threads;
there is no multiprocess data loader or Torch resource probing. Checkpoints save
atomically every 16 trees with exact resume identity. One writer holds the lock.
An observed training snapshot was about 6.6 GiB RSS at near four-core use; this
is a sampled value, not measured peak memory. Available disk was about 43 GiB.

The last retained bounded CREATE access attempt failed authentication. Current
remote jobs/assets are unknown, not assumed idle or verified. Local resources
were sufficient, so this experiment neither retried credentials nor submitted or
modified a remote job. The simulation project's jobs/files were not touched.

## Verification Coverage

- 48 scoped regression tests passed in 1.98 seconds before fitting. This includes
  the new 11-test feature module and existing risk/metric contracts; the earlier
  11-test invocation overlaps and is not counted again. No unchanged full legacy
  suite was rerun.
- All 36 source exclusions, checkpoint identities, 128-tree budgets, 118-column
  splits, matched target/draw/known hashes and target cutoffs were checked.
- The separate checker distinguishes the target/budget cutoff from the sampler's
  strata cutoff. Training always used the former; correcting that checker field
  did not change the registered experiment, targets or fits.
- 376776 original-unit query constraints, 103000 selected pointwise instances,
  180 scene reductions and all 30 prior summary rows pass separate arithmetic.
  This is a second computation by the same executor, not an independent reviewer.
- Five solver proposals fail the strict original-unit feasibility check and
  return CV. No budget is relaxed. They are not called optimal; exhaustive
  enumeration is not run. Earlier count-matched control failures remain failures.
- Decision arrays/records and aggregate replay are exact. The latter retains all
  39 rows, 27 paired comparisons with three subsets each, unknown-support counts
  and partial-future bounds.
- All 72 fixed feature/prediction unit probes are exact. This excludes the separate
  IMPTC precision repair and is not end-to-end controller invariance.
- The reporter, zero-CV diagnostic and plot scripts ran successfully. The rendered
  plot was visually checked for clipping, labels, legend and the 2% reference.
  Post-readout harm checks are explicitly descriptive and did not alter policy.

The [operation guide](operation_zh.md) contains the reproduction commands.
Private caches, query decisions, checkpoints and preview PNG remain excluded from
Git. Only code, configuration, aggregate reports and the aggregate SVG are shared.
The unrelated staged changes retain fingerprint
`c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323`.

No confirmation data were opened. No metric/seconds/true3D/foundation or
submission-readiness claim is made. Stage5C and SMC remain off.
