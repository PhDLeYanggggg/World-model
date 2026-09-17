# Limited Tool-Assisted Visual Review

This is an assistant's self-audit of private, input-selected contact sheets, not
an independent human annotation, person-identity certification or human-gold
label set. Images remain local and are not published in the repository.

The registered examples are the first, middle and last query in each selected
recording's request list. They were not chosen by loss or forecasting outcome.
Each sheet shows mapped current annotation boxes and eight past ego-centered
crops with retained support fractions. Strides 1 and 12 are still diagnostic.

| Recording | Displayed queries | Observations and limits |
| --- | --- | --- |
| bookstore/video0 | Agent2/frame6888/stride1; agent2/frame6972/stride12; agent235/frame9840/stride12 | Edge truncation remains explicit. The first two histories are low contrast and do not permit confident visual person identification at this resolution. The last history contains a small moving dark figure and partial edge support. Source flags alone are not a visibility certificate. |
| deathCircle/video2 | Agent2/frame7/stride1; agent2/frame84/stride12; agent30/frame420/stride12 | Resize mapping places the query boxes on the visible scene. Past crops contain road/sidewalk context and small dark actor-like details; the final building-corner example remains ambiguous. This does not certify every person or frame alignment. |
| hyang/video7 | Agent0/frame7/stride1; agent0/frame84/stride12; agent35/frame564/stride12 | Past path, moving shadows and small actor-like features are visible. Some crops are dominated by shadow/background. Retained pixels should not be equated with visible body detail. |

Nine displayed histories contain overlapping requests, so their 72 crop panels
are not 72 independent observations. The other 37 train recordings have automated
structural/pixel checks only, not this visual self-review. Immutable extraction
receipts retain `visual_review_status=not_reviewed` as their pre-review state;
this separate note records the limited subsequent review.

The aggregate projected-box-size audit is descriptive. Small boxes may restrict
available visual cues, but it cannot establish why earlier ETH/UCY models failed
or prove that a larger crop/encoder would improve forecasting. No training or
source-role admission follows from these examples. The source annotation histories
remain offline and often generated/interpolated, not sensor-as-of observations.
