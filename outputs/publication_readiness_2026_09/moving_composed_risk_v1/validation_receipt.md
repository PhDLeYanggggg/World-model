# Validation Receipt

## Observed Completion

| Phase | PID | Observed terminal exit | Seconds | Peak RSS bytes |
|---|---:|---:|---:|---:|
| Actual resume pilot | 98622 | 0 | 64.697539 | 6617432064 |
| Six fits and full treatment repeat | 98758 | 0 | 938.045702 | 6777159680 |
| Exposed-development evaluation | 1815 | 0 | 66.745309 | 5588910080 |
| Fitting and prediction replay | 2026 | 0 | 68.932640 | 5683314688 |
| Fitting-error attribution | 2273 | 0 | 33.205930 | 5695946752 |
| Attribution replay | 2488 | 0 | 33.116177 | 5739184128 |

Native arm64 PyTorch CPU4/inter-op1/workers0. No resource probing, NumPy model
fallback or HPC job. Historical CREATE receipts were reviewed as historical
evidence, not a live scheduler audit. Local resources were sufficient.

Both arms reproduce50+50 versus100 actual updates and predictions on the largest
context. Six final fits complete12,000 primary updates, plus2,000 exact full-refit
updates and400 pilot updates including repeats. Both neural encoders update.
All three independent controls exactly reproduce the preceding separate64
weights, optimizer, sampler, training traces and predictions. Input/target/
proposal/observed-mask hashes and query draws are verified against frozen origins.

All six initial/final fitting metrics, held predictions, selected actions and
cached-control metric recomputations replay exactly. All six fitting-error
decompositions replay exactly. The report generator and its regeneration exit0;
all31 pre-existing public experiment files remain byte-identical on regeneration.
Research-state numeric arrays, main contrast, runtime and held metrics match
the saved aggregate evidence.

## Focused Tests

36 tests pass in4.12seconds:

```sh
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_composed_risk.py tests/test_m3w_fixed_risk_readout.py tests/test_m3w_decoupled_risk.py tests/test_m3w_decoupled_risk_safety.py tests/test_m3w_full_objective_risk.py tests/test_m3w_annealed_risk.py tests/test_m3w_signed_decision.py tests/test_m3w_kinematic_risk.py tests/test_m3w_all_easy_gradient.py
```

Checks cover original unknown-label supervision, blocking-head gradients,
exact resumed fitting, interruption without saving half-applied optimizer
updates, target-mutation-invariant inference and future-field rejection.
The new synthetic readout test checks that all arms/actions are fixed before
outcome access, replay is exact, and a changed prediction cache is rejected.
Its first run failed because the temporary fixture root was outside the
project artifact root; the fixture was corrected, not production data rules.

The full historical suite was not rerun. Static/synthetic checks, actual
training and scientific performance are reported separately. No running phase
remained when the completion receipts were checked.

## Scientific Boundary

The internal fitting screen passes3/3, but the original scientific safety gate
does not:0/17 recording views satisfy full original risk/support. Worst defined
easy-net degradation is8.0378%, not within2%. The matched descriptive known
gain is only+0.01313pp. Completion bounds are not sampling confidence intervals.
Independent confirmation is not_run, and the original24-group primary is NA,
not passed. Old failures, missing-support requirements and empty-control NA
remain intact. No identity certification, new H, deployment upgrade or
submission-ready claim. Stage5C and SMC are disabled.

## Artifact Scope

Before this receipt,31 public Markdown/JSON files total1,021,168 bytes;
the largest is280,629 bytes. These are light aggregate metrics and reports,
not raw rows, checkpoints, caches, images, videos or third-party data.
Private artifacts remain outside Git. The unrelated staged-change sentinel is
unchanged: `c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323`.

Analysis SHA256: `46d0f8ef6a8ef9a0a45383ad0d1a4dc5b57079b0bf6023aec0a2187042efbced`.
Attribution SHA256: `feb04d6a7bfe6e8e31054a1aa971fbb798c83d40e016f18a11ad043f26388324`.
