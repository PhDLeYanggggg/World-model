# Final Preflight Guards Before Any Real Preparation Or Fitting

The first local registration draft precedes two additional recovery tests and
stronger completed-receipt/forecast identity checks. No real input preparation,
pilot,fitting or outcome scoring has run. The only private artifact is the
read-only CREATE observation receipt. Retain the first draft and its original
input-admission file;the active final definition is registration_v2.json.

Config,model,trainer and protocol hashes must match the draft byte-for-byte.
The runner now checks the completed receipt's4000step field and the forecast
checkpoint's identity,step and zero held-source draws. Tests additionally
exercise prepared-array idempotency and completed-receipt reuse. These are
pre-execution recovery guards,not outcome-selected model,loss,data,normalization,
split,threshold or primary changes. Commit the final definition before real work.
