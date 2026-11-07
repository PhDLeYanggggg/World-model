# Quantitative Risk-Shortfall Recipient Selection

Use the same 34 exposed Moving A recordings, 14 source groups,
three frozen forecaster seeds, 8/12 native steps, CV risk reference
and OLS4 comparator. Use the frozen source-excluded six-channel
cost heads. Nominate exactly one path per eligible agent using
the original predicted utility. This nomination is shared by all
recipient rules; no rule changes a nominated path.

At the original exact per-query switch count compare independent
gain ranking, support-first (one maximum estimated easy-reference
recipient, remaining by gain), and minimum predicted shortfall.
The latter chooses an assignment with positive predicted easy
reference if possible, minimizes the larger of zero, summed
predicted all excess and summed predicted easy excess, then
maximizes summed normalized gain. Stable ties use the original
agent order. This is exact over eligible recipients with one
fixed nomination each, not over arbitrary forecast combinations.
Apply the same existing predicted gate separately to each arm.
Matched proposal counts do not imply matched screened counts.

After the past-only freeze, score the original unmodified 2%
recording checks. The primary comparison is shortfall versus
support-first at the exact same switch counts. Report conservative
utility, complete passes, easy/support and paired path metrics,
with 3,000 paired source draws after seed averaging. An improvement
in one risk component is not an original rule pass.

Independently check whether the exact query budgets leave any
room: minimize each true-label risk component separately over
all recipients, first with the fixed nominations and then with
the full legal bank. The full-bank minima are optimistic and
can use different actions for each constraint. A positive
minimum proves infeasibility at these exact counts. A nonpositive
minimum does not certify a joint solution. Export only aggregate
statistics, no label-aware choices. No new external access,
threshold tuning, model promotion, Stage5C or SMC.
