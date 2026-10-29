# M3W Evidence Manuscript, Revision 6

This working revision integrates the completed multi-action cost, exact-record,
published regression-deferral and positive-support conditioning studies. It
does not replace or rewrite earlier manuscripts and failed experiments.

The manuscript is a substantive English research draft with methods, controlled
experiments, results, limitations, references and an LLM usage statement. It is
not a formatted eight-page CVPR submission, an independently confirmed method,
or a complete anonymized release. Private author administration is separate.

## Evidence Boundary

- New training/readouts are fresh_run, with endpoints frozen before scoring.
- This manuscript exporter is cached_verified aggregate reuse, not fresh science.
- All experiments retain three controller seeds and a frozen trajectory producer.
- Internal held recordings are exposed development, not independent test data.
- Full policies face every-record all/easy2%risk checks; empty risk is undefined.
- Count-matched controls concern utility, not safety.
- Aggregate archives contain numerical group summaries, not raw data or weights.
- Independent selection, calibration and confirmation remain unopened.

Read the [manuscript](manuscript.md), [complete comparison](../cost_decision_evidence_20261029.md),
[related-work position](../cost_decision_related_work_20261029.md),
and [operating guide](reproduction_zh.md).

## Exact Export Checks

From the repository root, after restoring the missing owned aggregate files:

```bash
.venv-pytorch/bin/python scripts/report_m3w_cost_decision_evidence.py --check
.venv-pytorch/bin/python scripts/build_m3w_evidence_manuscript_v6.py --check
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_cost_decision_evidence.py tests/test_m3w_evidence_manuscript_v6.py
```

The exporters verify pinned sources before reproducing their text. They do not
recompute confidence intervals, retrain models or infer forecasts. Fresh
inference additionally requires the ignored licensed/private data assets and
frozen checkpoints. The public aggregate package alone is not full end-to-end
reproducibility. Submission readiness remains false.

M3W remains a2.5Dtrajectory research scaffold. No metric/effective-second,
true3D,foundation,Stage5C execution orSMC claim is made.
