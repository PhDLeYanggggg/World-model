# Budget-Score Supervision:Development Results Addendum

## Design

We test whether directly supervising the deployed signed budget scores
H-0.02R and H_easy-0.02R_easy repairs a continuous cost selector.We retain
the five identified moment targets,ranking objective,head architecture,
462causal inputs,original sampling and preprocessing,three head seeds,
six fitting contexts and three frozen candidate banks.The two added
objectives use symmetric score MSE or4xweight on score underprediction.
All162models receive2,000updates;54original controls must reproduce prior
full scientific states.The original risk budget and decoder are unchanged.

This differs from the earlier score-only head,shared-reference architecture
and fixed teacher-region moment losses.It is a controlled objective
experiment,not a novelty claim for asymmetric regression or a safety theorem.
The detailed notation and observation estimand are in[paper_methods.md](paper_methods.md).

## Complete Fixed Results

| Objective | Physical-only known ADE gain | Old-neural known ADE gain | New-neural known ADE gain | New-policy recording risk support |
|---|---:|---:|---:|---:|
| Original | 1.946059% | 1.968657% | 1.893446% | 37/444 |
| Symmetric score | 3.041956% | 3.117461% | 3.015832% | 19/444 |
| Conservative score | 1.752622% | 1.851186% | 1.979966% | 41/444 |

The primary conservative-new minus original-new known contrast is
+0.086520pp,CI[-0.164383,+0.413791];its completion-lower interval crosses
zero.The symmetric arm's known increment is+1.122386pp,
CI[+0.494471,+1.865738],but it worsens selected recording risk.These
results do not support a safe complete-policy repair.

A prespecified secondary control restricts the same conservative new head
to physical actions without refitting.The actual neural-use known
increment is+0.209989pp,CI[+0.036449,+0.435968],and the recorded completion-
lower increment is+0.184456pp,CI[+0.013755,+0.405677].This isolates a small
useful neural contribution within that fitted head.It is not superiority
over every separately trained physical policy or independent generalization.

## Mechanism And Limits

Global legal-action score MSE improves for the symmetric arm while selected
score optimism and recording-risk violations worsen.The conservative arm
reduces some errors but retains negative selected-score bias and312known
easy-risk violations.Available full easy-reference mass is positive on
all148seed-deduplicated context/record views;undefined selected-easy risk
is not explained by complete absence of reference labels in those data.

All required localities and recordings remain.Intervals use3,000locality
resamples after context/seed averaging;overlapping windows and repeated
objective views are not independent samples.SOURCE is historically exposed
development data,and partial-row observed costs remain the stated estimand.
No independent selection/calibration/confirmation or controller-held
outcomes were read.The full original control and separate complete replay
receipts address computation,not scientific independence.

These results strengthen a diagnostic account of useful neural substitution
versus complete-policy safety;they do not complete the proposed method,
scene-joint evidence,anonymous reproduction release or submission package.
Not metric/seconds,human gold,true3D,foundation or a new dynamics training
claim.No Stage5C orSMC.
