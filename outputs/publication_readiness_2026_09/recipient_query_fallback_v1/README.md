# Query-Local Fallback at Matched Counts

Fresh causal-input readout of frozen source-excluded heads. Each
query uses its original past-only quota. A nomination is eligible
only if its predicted gain is positive and both predicted harm
inequalities pass for that path; eligibility and ranking are
independent of other queries. Both arms share the nominator and
eligible pool, keeping per-query counts identical. One ranks by
attainable recipient gain, the other by nominated-path gain.

The two arms choose identical actions in every one of 72
recording-seed views. Each makes **286 changes (0.137%)**, passes
**6/72** original risk checks, and has undefined easy risk in 63.
The direct conservative gain vs CV is **-0.000104 pp**
[-0.000228,-0.000020], and vs OLS4 **-12.129 pp**
[-21.779,-2.464], using 3,000 paired source-group draws.
This is post-failure diagnostic, not validation-selected deployment.
Empty and unsupported views are not counted as risk passes.

~~~sh
.venv-pytorch/bin/python scripts/run_m3w_recipient_query_fallback.py
~~~

Repeated execution verifies the existing frozen choices and scores
without refitting. The 2% rule, calibration and sealed confirmation
remain unchanged. No Stage5C or SMC.
[Results](analysis.json) | [Parent experiment](../recipient_feasibility_v1/README.md).
