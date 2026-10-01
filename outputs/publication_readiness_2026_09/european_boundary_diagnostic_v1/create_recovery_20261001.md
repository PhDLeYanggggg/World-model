# CREATE Replication Recovered and Verified

On1 October, a new read-only inspection of the existing job37602475 returned
`COMPLETED|0:0|00:01:47`. The completion heartbeat is dated30 September06:12:09UTC.
Earlier connection timeouts did not establish a job failure. No duplicate
submission, restart, cancellation or simulation-project change was made.

The existing collector checked all288 remote result hashes and the summary,
config, manifest and registration hashes. It retrieved only aggregate evidence.
Its cross-checks cover1,728 parent-risk values and three individual local/remote
proof packets. The preceding local reconstruction had already matched all288
exported packet bytes to the committed manifest.

Additional verification compares the **entire** local and CREATE summary:
all numeric, structural and Boolean fields agree using the existing strict
relative tolerance1e-10 / absolute tolerance1e-9. A first provenance-sensitive
comparison failed only because `result_source` correctly distinguishes local
execution from CREATE. The verifier asserts both exact expected provenance
strings and excludes only that field. No numeric tolerance was widened, and the
two summary files are not claimed byte-identical.

| Remote computation | Verified result |
|---|---:|
| Diagnostic groups | 288 (72 fitting, 216 transfer) |
| First pass | 52.97s |
| Exact remote replay | 45.03s |
| Scheduler elapsed | 107s |
| Independent native accounting checks | 64,512 |
| New parameter updates | 0 |

Reproduce the collected-evidence check locally:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/verify_m3w_boundary_create_parity.py
```

The source code and artifact hashes are bound in `create_parity_verification.json`
(SHA256 `17a7a5ed1e0d1676234dfffd77025e4780310ef1b3b9ac6894b866ee9dfa7175`).
The private collected scheduler receipt is required for this command; it contains
no newly collected row cache. The public compute receipt and summary are light
aggregate research evidence, not raw trajectories or checkpoints.

This note supersedes earlier **remote-status-only** `not_run`/unverified entries,
including historical sealed operation reports. Those artifacts are left intact.
Local and remote numeric replication is now verified; independent scientific
confirmation is still not_run. This was a frozen-model diagnostic, not training.
Its negative conditional-risk finding is unchanged. No deployment, metric,
seconds, human-gold, true3D, foundation, Stage5C execution or SMC claim.
