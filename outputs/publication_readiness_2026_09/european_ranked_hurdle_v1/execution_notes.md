# Execution Notes

Date:25September2026. Registration commit:a234cef2,pushed before real fitting.
Local native arm64 PyTorch,CPU4/inter-op1/workers0;single-process execution.
Previous turn was substantive progress:the completed matched-coverage diagnosis
motivated this one-factor training change. No remote job was submitted or altered.

| Phase | PID | UTC span | Outcome |
|---|---:|---|---|
|100-step fitting-only pilot |54521|11:05:30-11:05:33|Exit0;0.128s head fit,4.235s complete phase including load. |
|Resume,all training,replay,freeze decisions |54597|11:05:52-11:10:04|Exit0;254.194s complete phase. |
|First complete outcome evaluation |55000|11:10:15-11:11:00|Exit0;48.406s complete phase. |
|Full replay and separate arithmetic |55137|11:11:44-11:14:47|Exit0;184.990s complete phase. |

Sum of36 completed head fitting times,including the resumed pilot segment:
149.734s. This excludes loading,inference,checkpoint replay,metrics and reporting.
All72,000 registered updates completed. New neural forecaster training=0.
The real CPU sample showed PID54597 progressing at approximately105% CPU and
6.34GiB resident memory during execution;this is one observation,not peak usage.
Disk was approximately26GiB free;the registered below10GiB guard did not trigger.

Each head uses atomic checkpoint replacement every200 steps,optimizer and RNG
state,training draws,identity bindings and losses. The pilot was resumed. No
experiment crashed,no timeout downgrade occurred,and no required process remains
running. Heartbeat and append-only events are local at:
`data/stage_cvpr2027_experiments/european_ranked_hurdle_v1/`.

The analysis SHA256 is
`9f51a2f9e3a1ee977de196b244a63063ca82786c758b981ef0eb4b69e485b8f2`.
Checkpoint hashes and sampled replay scope are in[head replay](head_replay.json).
Full metrics and separate scalar-sort/coordinate/bootstrap implementation agree;
the[verification receipt](verification.json) binds them to unchanged source code.
[Completion checks](completion_checks.json) record all terminal PIDs and234
passing tests in37 scoped files,not a full legacy-suite run.

The seven new tests cover pair locality boundaries,ordering direction,undefined
and tied labels,zero-reference positive harm,margin weighting,exact zero-auxiliary
control equivalence,and interrupted/resumed fitting with bound settings. The
test-first missing-module failure was expected before implementation,not a real
training failure. No prior valid implementation or evidence was removed.

CREATE was unnecessary for this small-head matched experiment. No live CREATE
availability claim is made;previous scheduler observations remain timestamped
history,not proof of current remote status. The underlying large forecasts and
cross-fitted feature lineage were reused with hashes,not silently retrained.

Only code,configuration,reports,aggregate metrics and SVG scientific figures are
published. Raw detector tracks,cache,feature/history/latent stores,checkpoint
binaries,row-level decisions and PNG previews stay local. The private full
analysis is ignored. Unrelated pre-existing staged work is preserved.

The frozen control's internal helper alias is `product`;the new ranked head's
alias is `hurdle`. Public tables rename them control/ranked. There was no
product-MSE training arm this round. This explicit compatibility mapping changes
no rule,score or metric.

All data are opened development,not independent final testing. Released detector
pixels,obs8/pred12 rawstride12;no seconds/metric/physical-safety claim. Reserved
roles and deployment remain unchanged. Stage5C and SMC stay disabled.
