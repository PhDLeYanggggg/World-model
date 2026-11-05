# Completion And Verification Receipt

- Real pilot PID87463: terminal exit0;400updates including repeats, exactresume.
- Real training PID87626: terminal exit0;521.016724seconds, peakRSS9,240,248,320bytes.
- Six new heads:12,000primaryupdates; exact full refit:2,000additionalupdates.
- All six controls match target/proposal/draw/preprocessing hashes; only input354removed.
- Evaluation PID88461: terminal exit0,48.023267seconds.
- Replay PID88560: terminal exit0,46.734500seconds; all3contexts exact.
- Old cached-control actions verified, metrics and fitting diagnostics recomputed exactly.
- Report generation and final replay: terminal exit0. All31existing public
  evidence files byte-identical before/after the final replay. This receipt
  is added afterward and is not one of those31files.
- Main/secondary state metrics checked against saved analysis; exactSHA256
  `7bfb8ed0a807c44f9c7efe8bf7844f1ddd65c79761854eaf24cc2109255549ca`.
- Final focused suite:31passed in3.06seconds. Full historical suite notrerun.
- Original2%risk/easy-net/support rules unchanged; no identitycertificate.
- Main and secondary expansion screens false; full risk/support0/17and1/17.
- Original24group primaryNA. Independentconfirmationnot_run; reserved data closed.
- No deploymentchange, submission-ready claim, Stage5Cexecution orSMC.
- Training checkpoints and27MiBprivate artifacts remainlocal; public evidence
  is light JSON/Markdown. Unrelated staged changes remain excluded from commits.

Focused validation command:

```sh
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_scale_ablation_reporting.py tests/test_m3w_scale_ablated_risk.py tests/test_m3w_feature_support_audit.py tests/test_m3w_signed_decision.py tests/test_m3w_signed_decision_reporting.py tests/test_m3w_kinematic_risk.py tests/test_m3w_direct_tail_risk.py tests/test_m3w_admission_priority.py tests/test_m3w_source_error_priority.py
```

The scientific conclusion is a local utility effect with failed safety, not
independent world-model success. Deployment and the original failures remain.
