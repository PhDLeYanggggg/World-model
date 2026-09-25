# Execution and Reproduction Record

Date: 25 September 2026. Registration fb876710 was pushed before actual fitting.
Previous turn was progress: ranked-hurdle training and negative findings were
completed, verified and pushed as aa811b10. This turn changes pair formation.

| Phase | PID | UTC span | Observed outcome |
|---|---:|---|---|
| Bound-asset preparation | 56775 | 11:30:46 | Exit 0; 2.096 seconds including load. |
| Fitting-only support audit and 100-step pilot | 56861 | 11:31:29-11:32:05 | Exit 0; 39.050 seconds including load/audit; pilot fit 0.242 seconds. |
| Resume, all fitting, replay and decision freeze | 56949 | 11:32:42-11:36:07 | Exit 0; 207.818 seconds including load. |
| First complete evaluation | 57362 | 11:36:36-11:37:19 | Exit 0; 45.881 seconds including load. |
| Full replay plus separate arithmetic | 57466 | 11:37:55-11:40:50 | Exit 0; 177.317 seconds including load. |

All 36 heads completed 2,000 updates; 72,000 total. Summed reported fitting time
is 111.802 seconds including the pilot segment, excluding preparation, forecast
inference, replay, evaluation and reporting. Different scheduling means this is
not a controlled speedup benchmark against the preceding fit.

Native arm64 PyTorch CPU4/inter-op1/workers0, with no DataLoader multiprocessing
or Torch resource probing. The actual training process was observed progressing
at 143.5% CPU and 5,738,928 KiB RSS (about 5.47 GiB); this is a sample, not peak
memory. Available disk was about 26 GiB, above the unchanged 10 GiB guard.
All required phases finished; no restart from scratch, no hidden matrix reduction
and no required training/evaluation process remains running.

Atomic checkpoints contain optimizer, RNG, sample draws, original and fixed-batch
losses, identity bindings and pair totals. The real pilot was resumed. Public
code and aggregate receipts are available; checkpoint binaries and row-level
arrays remain in the private experiment directory:
`data/stage_cvpr2027_experiments/european_supported_pairs_v1/`.

The full analysis SHA256 is
`523e6b9dec44cbe2a50922e20bc9540565b1df410fee96ca98f1d2544e2653d9`.
All 36 replays use the first 4,096 excluded-index rows per head, not a random
sample. Full 216-view metrics also reproduce, including exact old controls.
A separate scalar-sort/coordinate/bootstrap implementation validates the result;
it shares the frozen inputs and is not independent research confirmation.

The new targeted tests cover sparse support, locality/tie boundaries, invalid
labels, zero-reference positive harm, exact loss/gradient equivalence, unchanged
fit/sample draws where pairing has no effect, fixed-batch diagnostics and resume.
Two additional tests check the constructed estimand counterexample. Initial
missing-module failures occurred before implementation, not during real fitting.
The final scope has 247 passing tests in 39 files. The full legacy suite was not
rerun because its historical checks write unrelated reports and are non-hermetic.

Local assets and matching remote Git commits were verified. CREATE was not used:
this matched head experiment fits local resources. Existing same-day CREATE and
venue-policy records were read, not refreshed; no live remote availability or
new policy-verification claim is made. No remote job was changed or duplicated.

Only explicit code, configs, reports, aggregate metrics and SVG figures are
committed. Raw data, private caches, full analysis, checkpoints and PNG previews
remain local. The pre-existing unrelated staged work is preserved. Reserved roles
remain closed; deployment, Stage5C and SMC remain unchanged.
