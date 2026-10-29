# Signed Easy-Degradation Implementation Amendment

Status: fresh_run synthetic regression repair; not a scientific model result.

The frozen prospective preflight v1 incorrectly rejects every negative value
of `easy_degradation_upper`. With completely known outcomes, a negative net
degradation is a valid improvement. Positive harm/reference risk and signed net
degradation are different quantities. Their common upper budget of 2% does not
make their lower domains identical.

The new `src/evaluation/m3w_policy_contract_v2.py` keeps the original signed value
in its returned evidence and preserves caller inputs. For the budget comparison
only, it uses the equivalent `max(0, degradation) <= .02`. Negative positive-harm
ratios, undefined support, nonfinite values, insufficient utility and violations
of the 2% budget still fail. It does not change the population-safety claim.

V1 code, registered hashes and old verdicts are unchanged. No failed scientific
result is re-scored or promoted. The current shared-TRAIN experiment uses its
separately registered finite-completion calculation, which already accepts
negative known-label degradation; its training, readout and gates are unchanged.
Use v2 for new consumers of the generic prospective advancement helper.

```bash
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_policy_contract_preflight.py tests/test_m3w_policy_contract_v2.py
```

This verifies implementation semantics, not deployability. Stage5C and SMC stay
disabled; no independent evaluation is opened by this amendment.
