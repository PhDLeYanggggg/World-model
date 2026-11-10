# OpenDD: how much can one edit exchange recover?

I kept the forecasts, 780 realized edits, recording-level movement caps,
chronological credits and original 2% rule fixed. I asked how much a
hindsight search could recover by changing at most one or two of the
frozen allocator's edits. An exchange removes a chosen forecast and
spends that edit at a different query, within the same nested menu.

| Allowed exchanges | Strictly positive / 606 | Above 0.10 pp | Above 0.50 pp | Above 1.00 pp |
| --- | ---: | ---: | ---: | ---: |
| None | 60 | 21 | 1 | 0 |
| At most one | 178--183 | 68--69 | 4 | 0 |
| At most two | 205--207 | 96 | 7 | 1 |
| Unrestricted | 210--212 | 109 | 18 | 1 |

The left end counts plans verified with the original trajectory scorer;
the right end adds cases not excluded by the upper bound. These are not
confidence intervals. Each gain floor must be exceeded against both CV
and OLS4 using the original conservative score and CV denominator. The
extra floors are descriptive, not proposed release thresholds.

One exchange adds 118 verified passes (equal-location +11.93 percentage
points, 95% location bootstrap [+4.90, +18.72]). Among those, 107 had
failed the original risk checks, including unavailable risk estimates,
and 11 had passed risk but not utility. At least 29 of the unrestricted
passing views provably require more than one exchange. The opportunity
is therefore often local, but large gains are rare at this budget. A
past-only rule still has to identify the correct exchange; hindsight
does not establish that it can.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_opendd_exchange_frontier.py
.venv-pytorch/bin/python scripts/report_m3w_opendd_exchange_failures.py
```

The fresh audit ran on native arm64 with zero optimizer updates, using
the same seven locations and three seeds (202 recordings, 606 views).
Resume validates private witnesses against the original scorer. See the
[protocol](protocol.md), [registration](registration.json), [readout](readout.json)
and [failure decomposition](failure_decomposition.json). No external
labels were reopened and no policy was replaced.
