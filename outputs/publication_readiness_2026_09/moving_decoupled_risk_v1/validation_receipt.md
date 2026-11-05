# Validation Receipt

## Verified Scope

- Native arm64 PyTorch CPU4/inter-op1/workers0, no resource probing or NumPy model fallback.
- Pilot PID95467, observed terminal exit0:84.815638s, peak6,640,500,736 bytes.
- Training PID95634, observed terminal exit0:1,059.521734s, peak7,916,552,192 bytes.
- Nine fixed fits /18,000 actual updates; first separated fit repeated for2,000 exact updates.
- Largest-context50+50 versus100 actual-update pilot matches for all3 architectures;600 pilot updates including repeats.
- Input/target/observed-mask/proposal hashes and query draws match frozen origins.
- Evaluate PID97225, exit0: only records not_run after the prespecified fitting screen fails2/3, not a held prediction pass.
- Verify PID97261, exit0: all9 initial/final complete-fitting metrics replay exactly.
- Attribution PID97310 and replay PID97358, both exit0: all9 decompositions exact.
- Report generator and regeneration both exit0; all33 pre-existing public experiment files remain byte-identical after regeneration.
- README/research-state counts, runtime, analysis hashes and separate-model numeric arrays agree with saved results.

## Checks And Limitations

29 focused tests passed in3.48seconds:

```sh
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_decoupled_risk.py tests/test_m3w_decoupled_risk_safety.py tests/test_m3w_full_objective_risk.py tests/test_m3w_annealed_risk.py tests/test_m3w_signed_decision.py tests/test_m3w_kinematic_risk.py tests/test_m3w_all_easy_gradient.py
```

The first new bound-gradient test failed because its synthetic fixture had no
unknown-label legal action. The fixture was corrected before registration and
real training; actual labels and supervision rules were not changed. The final
tests include preserved unknown-label gradients, exact resumed fits, future-field
rejection, target-mutation-invariant inference, and interrupted-optimizer recovery.
The failed-fitting-screen test verifies that no held packet is fetched.

The full historical suite was not rerun. Existing registrations, source lineage,
warm checkpoints and cached feature/target bindings are **cached_verified**,
not new independent data. Model fits, fitting metrics and attribution are
**fresh_run**. Additional held predictions are **not_run** with the registered
reason. The held-success branch of this new runner did not execute on real data;
no claim of its real-data validation is made.

Original2% risk/support, empty-control NA, old failures and the unsupported
24-group primary are unchanged. No identity certificate, independent test,
deployment upgrade or submission-ready claim. Stage5C/SMC are disabled.

## Artifact Boundaries

Before this receipt, the public directory contained33 Markdown/JSON files,
278,029 bytes total, largest115,457 bytes. No row cache, checkpoint, video,
image, raw data or third-party asset is included. The unrelated staged-change
sentinel remains `c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323`.

Analysis SHA256: `a42c79de2f1657c6562c3bb8c05e324ae4c7ad7d882e9d1353ff92f04f078dbf`.
Attribution SHA256: `fd23aebe54c4dbc06f8b67e8c003f9a9f09855f37dd6947efbfc95f0296714c6`.
