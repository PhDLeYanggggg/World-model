# v202: matched forecast backbones and a shorter development section

The main paper now compares CV, OLS4, the completion-trained forecaster,
EqMotion and Social-STGCNN on the matched 500-recording OpenDD vehicle task.
Both graph backbones use the same loss, input roster and fit budget. This is
a deterministic adaptation, not a ranking on either original benchmark.
Relative to the matched EqMotion control, Social-STGCNN gives -16.66 pp
conservative gain (95% location interval [-18.96, -14.31]); this is a direct
paired contrast, not the difference between the two CV gain columns.
Adding neighbors to Social-STGCNN gives +23.60 pp on observed steps, but
-1.15 pp [-4.54, +2.24] on the conservative contrast.

I condensed the vehicle angle and calibration ablations to the fixed-budget
nomination finding and moved the discussion of individual fits out of the
main text. The older draft and experiment receipts retain their details.
The Songdo section points to three retrospective cases in the anonymous
supplement. Nothing in the calibration, test roster or release rule changed.

| Artifact | What it establishes |
|---|---|
| [OpenDD training and direct contrasts](../opendd_vehicle_stgcnn_v1/README.md) | The matched architecture experiment was run previously; v202 checks its frozen inputs and renders the result. |
| [Songdo case evidence](../songdo_failure_cases_v1/README.md) | Three illustrations from previously scored flights, not additional confirmation. |
| [Build receipt](receipt.json) | Seven-page PDF; title, abstract and LLM statement unchanged. |
| [Anonymous replay receipt](../songdo_failure_cases_v1/review_receipt.json) | v34 was originally paired with v201. The unchanged archive contains the controls and cases cited in v202; it replays summaries, not raw training. |

PDF and source ZIP are local in
`data/stage_cvpr2027_experiments/evidence_manuscript_v202_focused_evidence/`.
The [Chinese reproduction guide](reproduction_zh.md) covers native runtime,
checkpoint continuation and CREATE result checks. Nothing was uploaded.

## Remaining evidence, in order

1. A positive effect from joint selection at equal budget and under the
   original risk rule remains unproven. Neither frozen external result
   supports deployment of the joint rule. OpenDD architecture results are
   development comparisons, not new confirmation.
2. The frozen Athens proposals have an outcome-free readout rehearsal.
   The real labels stay closed until the acquisition-date roles are agreed.
   Keep the original 2% checks and separate the matched from released results.
3. Reconcile the enrollment receipt and private author fields separately
   from the paper assets. CVPR's deadlines are Nov 10 (registration), Nov 16
   (paper), and Nov 23 (supplement), all AoE. See the
   [official calendar](https://cvpr.thecvf.com/Conferences/2027/Dates) and
   [author guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines).
   This build does not establish submission readiness.
