# Execution and Reproduction

## Material Passport

2026-09-22. Native arm64 local execution in `.venv-pytorch`, Torch 2.12.0,
CPU four compute threads, one inter-op thread, zero DataLoader workers. This
experiment is small downstream inference/analysis, not a new full neural fit.
No runtime fallback, new CREATE job, data-role change, Stage5C execution or SMC.

## Source and Process Evidence

- Registration commit: `2744393872ab0d098a1898cc9d1df7be5c1a6fea`, pushed before readout.
- Preflight: 1,110 source bindings verified; terminal exit 0.
- Fixed evaluation session `61708`: terminal exit 0.
- Checkpoint replay session `68684`: terminal exit 0, 36 endpoints and 1,581,804 scores.
- Separate-formula session `70006`: terminal exit 0, 36 choices and 288 scene reductions.
- Veto diagnosis session `78525`: terminal exit 0; no fitting or policy change.
- Exact diagnosis replay session `70979`: terminal exit 0; immutable output matches.
- Analysis SHA256: `1883a8501170b81b42b32dee1b4820a0396bb7e80f1f84d680e46fad864ef15a`.
- Veto diagnosis SHA256: `1b18ddf55e14273eb547595ac13842d76c1724d05efd34913dc3487464ec1a02`.

The prior evaluation/replay inspected verified cached checkpoints. The veto audit
is new arithmetic on frozen decisions and outcomes, not new training. It checks
disjoint veto causes, accepted/rejected partitions, complete-outcome net-benefit
conservation and all previous conditional summary values. Its AUROC implementation
is tested against explicit pair comparisons, including ties and absent classes.

The 38 unchanged experiment tests are reused from the recorded passing run.
Ten new veto-audit tests pass in 0.12 seconds. Full historical `tests/` is not
claimed passing: it was not rerun. No raw trajectory rows or checkpoints are
included in the result commit. Light aggregate JSON includes source fingerprints.

## Commands

Run from the repository root. Existing source data and checkpoints are required;
the Git repository alone is not a redistributable dataset/model bundle.

```text
.venv-pytorch/bin/python scripts/run_m3w_cross_objective_review.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_cross_objective_review.py --evaluate
.venv-pytorch/bin/python scripts/run_m3w_cross_objective_review.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_cross_objective_review.py
.venv-pytorch/bin/python scripts/audit_m3w_review_veto.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_review_veto_audit.py
```

Evaluation uses an exclusive process lock and writes PID/state heartbeat and
event logs to ignored `data/stage_cvpr2027_experiments/cross_objective_review_v1`.
Results are immutable: replay must agree rather than overwrite changed evidence.
There is no new optimizer to resume. If interrupted, inspect the specific process
first, retain completed archives, then rerun the same fixed command after terminal
status is established. Never change hashes to force incompatible cache reuse.

## Resources and Remaining Access

The local volume has about 53 GiB free. No additional large data is materialized.
GitHub main was freshly checked at registration commit before this results update.
The local SSH config still has no CREATE alias; prior CREATE authentication and
project-path issues are unresolved. This turn did not query a remote scheduler
or inspect remote assets, so their existence/status remains unknown, not absent.
No duplicate cluster job was submitted for this inexpensive local analysis.

All required evaluation, replay and diagnostic processes are terminal; there is
no unfinished local training run belonging to this experiment.

Independent-data acquisition and role questions remain pending. Neither elapsed
time nor a source-development bootstrap can replace those decisions. All claims
remain pixel/annotation-step development evidence, not metric/seconds or formal
safety guarantees.
