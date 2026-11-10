# Supplement: how much can one edit exchange recover?

Extract the archive and run `python3 -I -B -S verify.py`. The script uses
only Python's standard library. It checks the earlier decision evidence,
then recomputes the exchange frontier and paired location bootstrap from
anonymized score aggregates. It does not rerun the trajectory scorer,
mixed-integer search, model training or external outcome evaluation.

Every comparison uses 606 OpenDD development views, the same 780 edits,
chronological credits, per-recording movement caps and original 2% rule.
An exchange removes an edited path from the frozen-score allocation and
spends the same edit at another query. The search sees outcomes, so the
frontier measures opportunity rather than online policy performance.

| Allowed exchanges | Strictly positive | Above 0.10 pp | Above 0.50 pp | Above 1.00 pp |
| --- | ---: | ---: | ---: | ---: |
| None | 60 | 21 | 1 | 0 |
| At most one | 178--183 | 68--69 | 4 | 0 |
| At most two | 205--207 | 96 | 7 | 1 |
| Unrestricted | 210--212 | 109 | 18 | 1 |

Ranges here are verified lower counts and upper counts not excluded by
the solver, not confidence intervals. The paired location bootstrap is
reported separately. The additional gain floors are descriptive checks
only; they do not replace the original release criteria.
