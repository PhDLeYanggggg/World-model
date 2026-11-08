# Recovered CREATE Accounting

A fresh, read-only query using the previously verified connection from
the simulation project returned successful completion for all four
named M3W jobs:

| Job | Role | Terminal state | Exit | Elapsed |
|---|---|---|---|---|
| 37835856_0 | Recovery | COMPLETED | 0:0 | 4m 09s |
| 37835859 | Join | COMPLETED | 0:0 | 18s |
| 37837636 | Portability | COMPLETED | 0:0 | 3m 40s |
| 38287008 | Frozen payload verification | COMPLETED | 0:0 | 6s |

The remote training freeze SHA-256 matches the archived local freeze
exactly. This resolves the historical accounting uncertainty, not
the scientific failure. The [previous full readout](../european_easy_harm_deviance_v1/readout_verified_v3/conclusions.md)
already analyzed 72 development views across 12 localities and did not
qualify the candidate. That report and its old access history remain
unchanged. I did not rerun inference, download checkpoint weights,
or inspect the simulation project's scientific data.

No job was submitted, restarted or cancelled; no free GPU capacity or
future access is inferred. A later run should recheck the named jobs
and allocation before submitting. [Structured receipt](receipt.json).
