# Matched Recording Allocation and Causal Credit Carry

Keep the frozen preference ranker, CV harm screen, legal candidate library,
and the single chosen path per row fixed. Use the same nine fitting-source
rotations by fold and seed as the parent result; do not train or tune on
the held fitting recordings. Analytic-only calibration and source-excluded
held donors are unchanged.

The local allowance is the count given by the earlier nominate-first
screen, capped at eight, observable at the current query. The old
preference policy spends exactly that allowance with its ranked path
per row. For the two offline diagnostics, allow the count to move
between queries in the same recording, keeping exactly the same total
and the same cap of eight. Use gain alone or a soft risk price. The
gain score is the frozen normalized gain times the causal speed scale,
divided by query size. The priced score subtracts twice the positive
parts of the two frozen excesses, scaled by the causal speed and
divided by the median query size in the analytic calibration role.
Sort each query's best eight then globally rank those candidates.
These are retrospective allocation diagnostics over predicted costs,
not causal online policies or constrained risk certificates.

For streaming, use the median priced priority of eligible analytic
calibration rows as the minimum priority. Both streaming variants use
the same current allowance and same priority. One drops unused credits;
the other carries up to eight forward for up to four nonempty queries,
spending oldest credit first. No future queries, query count or later
labels are needed online. Streaming counts can differ between arms;
only the offline comparisons exactly match each recording total.
Apply the original 15-setting risk-screen ladder to the carry arm on
calibration. Require every original risk check and positive lower
utility against both references, else return CV. The fixed relaxed
diagnostic is not a threshold selection on held outcomes.

Freeze all held choices before scoring. Report original passes and
undefined-to-defined transitions, paid past credits, paired full-grid
lower gains with 3,000 source bootstrap draws after averaging seeds,
and performance against CV and OLS4. No A outcome, independent
confirmation, or change to the original risk rules or external split.
