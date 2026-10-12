# Recovery Completed; Scientific Evaluation Still Pending

## Verified Training Progress

Result source: fresh_run accounting/log observations at16:25-16:26 UTC on7 October.

- Owned recovery37835856_0 completed with exit0:0 in4m09s.
- Its batch peak RSS was1,986,104KiB, approximately1.89GiB, below the8GiB reservation.
- Its error log was empty. The final owned heartbeat reports shard_complete and36fit receipts.
- The recovery contract reuses17 diagnostic quadratic controls and trains17
  missing candidate heads. Two previously accepted shard-zero fits are retained.
  This is not144 fresh fits, and the adopted states are not labelled retraining.
- The original node and scientific training contract were unchanged.

The verification join37835859 remained PENDING/Priority in that observation,
with its dependency satisfied. No training_freeze.json was then present.
Consequently, neither a complete144-fit collection nor the scientific readout
is claimed. The54 historical-exact plus18 original-replay control admission
remains explicit; historical bitwise reproduction is not complete.

## Access Change After Session Resumption

On9 October local time (8 October UTC), resumed and fresh SSH observations were
closed at CREATE access verification. The portal was readable but showed new
pending source approvals. Their provenance was not established, so none was
approved or rejected. No access setting, job or checkpoint was changed.

This is an observation blocker, not training failure. It does not revoke the
earlier completed accounting record and does not prove the join is still pending.
Current join and diagnostic outcomes are unknown until a fresh authenticated
read succeeds. Source IPs and the portal screenshot remain outside public Git.

Fresh read-only attempts at04:10:45 UTC on9 October and01:04:20 UTC on12 October
record the same access-approval connection closure. The same blocker has now
recurred across three goal turns. No sources were approved, jobs resubmitted or
remote settings changed. All independent local handover work is saved; continuing
the already-frozen scientific experiment requires restoring verified access.

The last successfully observed training completion remains valid. Do not infer
the present status of the join or portability job from that old observation.
The two timestamped transport receipts contain no raw login banner or source IP.

The read-only observer now records transport errors separately from scheduler
and scientific status. It omits raw login banners, does not retry automatically,
and never interprets a portal banner alone as proof of authentication failure.
Fifty-six focused transport/scheduling tests pass; the full legacy suite was
not rerun. These are engineering checks, not evidence of predictive improvement.

## Resume Sequence

Use the existing native arm64 environment; do not submit duplicate training:

```sh
.venv-pytorch/bin/python -m scripts.observe_m3w_recovery_resilient
```

Only after the existing join is authoritatively completed0:0 and the freeze exists:

```sh
.venv-pytorch/bin/python -m scripts.manage_m3w_easy_harm_deviance collect --phase train
```

Inspect and hash-verify the144 unique2,000-step fit receipts and the registered
54/18 control counts. Commit only the collected lightweight evidence, preserving
unrelated staged work. Then run the unchanged registered readout:

```sh
.venv-pytorch/bin/python -m scripts.run_m3w_easy_harm_readout_v2 preflight
.venv-pytorch/bin/python -m scripts.run_m3w_easy_harm_readout_v2 run
```

If a prior readout is incomplete, first verify no process holds its lock and
inspect saved group hashes; only then use its explicit --resume. Do not overwrite
a completed readout or reselect thresholds/checkpoints. The independent roles
remain closed. Stage5C and SMC remain off. No metric/seconds, physical-safety,
true3D, foundation-model or submission-readiness claim follows from this update.

Evidence: resilient_observation_20261007T162548.json and
recovery_complete_join_observation_20261007T1627Z.json. Embedded observation
timestamps, not filing-name suffixes, determine chronology.
