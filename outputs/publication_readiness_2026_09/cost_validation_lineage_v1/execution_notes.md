# Execution And Reproduction

## What Ran

Local native `.venv-pytorch/bin/python`, read-only input audit. No Torch import,
checkpoint deserialization, model fitting, forecast inference or target-array
member access in the audit runner. It reads only `identities_json` from the OOF
archives and hashes the complete archive bytes. Hashing stored future-target
bytes is not using those targets as inputs or computing a new label readout.

142 source files are hash-bound. Parent Transformer fit report:
`5d7e63098b5705e3c50e9c2e01aa3ac321edc607bfa7f84f24465e1b9429b024`.
Parent EqMotion fit report:
`2854a2fd50a55ccdd0af7ec2d879deedd3f6212b270f68963bbd536037b0e9a5`.
Historical model source is verified in its existing isolated mirror; the
working training/evaluation modules are not changed or installed into this run.

Fresh audit: PID 51067, exit 0, 4.388 seconds. Exact completed replay: PID 51103,
exit 0, 4.235 seconds. Both check all six family/seed groups. No long-running
training, remote job or checkpoint recovery was required. All required process
sessions finished; no background training is claimed.

Analysis SHA-256:
`b10544c99c1ff80a0e4a4920bfa26edb96c8401073a4dff9a188f800987cc2a4`.
The analysis includes its config and implementation hashes. Resume rechecks
the source bytes and reconstructs the same analysis rather than trusting an
existing completion flag.

## Commands

From the repository root, the existing completed result is verified with:

```bash
.venv-pytorch/bin/python scripts/audit_m3w_cost_validation_lineage.py --resume
.venv-pytorch/bin/python -m pytest tests/test_m3w_cost_validation_lineage.py tests/test_m3w_experiment_contract.py tests/test_m3w_risk_calibration.py -q
```

A first run without the published `analysis.json` uses the same audit command
without `--resume`. Input assets are local and intentionally not committed.
The runner fails on missing or changed inputs; it does not download replacements
or bypass frozen source hashes. It will not overwrite an existing audit silently.

67 scoped tests passed in 18.38 seconds, no skips. These include synthetic
contract and risk-calibration fixtures, not new real-data training. The full
historical suite, including tests that write old reports, was not rerun.

A separate reduction of the saved detail checks 18 producer passes, 36 direct
head-fold rejections, 18 indirect reuse failures, 36 nested requirements and
144 rejected pool candidates. The research-state analysis digest matches.
264 local links in the overview, results ledger and new notes resolve.
These engineering checks do not establish prediction improvement.

## Scope Limits

- Complete checks for the configured v6 artifact pool, not every project model.
- Verification of declared lineage and bound bytes, not proof of undocumented
  historical use or calibration independence.
- New pre-fit rejection helper, not retroactive validation of any old head.
- New audit with verified cached inputs, not fresh training or new test results.
- Existing thresholds, evaluation metric and scientific data roles unchanged.
- No Stage5C, SMC, physical-safety certificate, metric or seconds-level claim.
