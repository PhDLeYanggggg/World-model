# Fixed-budget direction and timing: scalar replay

Run `python -I -B -S verify.py` from this extracted archive. No dependencies,
credentials, raw labels, forecast arrays, models or repository are required.

The payload groups anonymous scenarios by city and preserves three seed
views per scenario. It includes the fixed 3,000 city bootstrap draws.
The verifier recomputes the all-view and complete-label counts, and paired
intervals after averaging seeds within scenarios. Intervals are fractions
of views, so multiply by 100 for percentage points.

The payload contains audited eligibility flags, not raw forecast or risk
arithmetic. It is a development-only comparison and does not establish
independent confirmation, release qualification or model training replay.
