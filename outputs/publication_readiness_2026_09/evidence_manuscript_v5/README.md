# M3W Evidence Manuscript, Revision 5

This revision incorporates the completed easy-harm deviance comparison rather
than leaving it described as pending. It keeps the earlier temporal experiment,
TRAIN diagnosis, control-reproduction limitations and all unfavorable comparisons.
Revision 4 and its isolated aggregate reproduction package remain unchanged.

## Main Finding

Changing the fifth easy-harm moment loss alone is insufficient. The candidate
loses to quadratic in full paired lower utility by 0.902716 percentage points
(nominal 95% locality interval: -1.775623 to -0.060409). Its positive comparison
to the original forest is not a pass against the stronger control. Known easy
risk fails in 59/72 views and the completion-upper risk bound fails in 71/72.
The risk is positive selected easy harm/reference error, not net easy ADE change.

The descriptive follow-up finds harm underprediction in all 72 candidate views.
Its median predicted/actual selected-harm ratio is 0.048558. This narrows the
failure description but does not identify a unique causal mechanism or prove
that a decision-focused loss would repair it. Matched decision-learning training
remains not_run. Independent selection, calibration and confirmation remain closed.

## Read and Reproduce

- [English manuscript](manuscript.md)
- [Eight evidence tables](tables.md)
- [Source hashes and numerical evidence](evidence.json)
- [Complete comparison](../european_easy_harm_deviance_v1/readout_verified_v3/conclusions.md)
- [Selected-risk diagnosis](../european_easy_harm_deviance_v1/readout_verified_v3/selected_harm_diagnosis.md)
- [Chinese operating guide](../local_create_runbook_zh.md)

From the repository root:

```bash
.venv-pytorch/bin/python -m scripts.build_m3w_evidence_manuscript_v5 --check
.venv-pytorch/bin/python -m pytest tests/test_m3w_evidence_manuscript_v5.py tests/test_m3w_easy_harm_selected_diagnosis.py tests/test_m3w_easy_harm_readout_v3_report.py -q
```

The exporter verifies pinned sources and group hashes, then reconstructs the
manuscript from cached verified results. It does not retrain, infer trajectories
or compute another bootstrap. The separate readout reporter can recompute the
paired-locality summary from the completed group aggregates. Three head seeds
are not three independently trained forecasters; repeated windows are not
independent observations.

## Submission Boundary

An LLM usage statement is included before the references, reflecting assistance
throughout the workflow rather than claiming grammar-only use. The draft is not
an anonymous eight-page venue submission. Independent confirmation, matched
decision-focused controls, scientific review and a complete submission package
remain incomplete. The revision-four standalone archive has not been upgraded
to reproduce these new results. No paper has been submitted.

M3W remains a 2.5D trajectory research scaffold, not true 3D or a foundation
world model. No new metric or effective-seconds claim is supported. Stage5C and
SMC remain disabled. Formal submission still requires author confirmation.
