# Does Observed Peer Motion Identify Useful Waits?

I compared self history, aligned histories of the nearest two observed
peers, and a wrong-focal control from the same query. The control does not
use a later frame or another recording. All models have the same head,
causal opportunity summaries, ledger state, labels, thresholds and splits;
the self arm zeroes the peer features. Nominations and counts used to
prepare each fitting donor exclude both that donor and the outer held
location. Past histories contain eight native annotation positions.

The positive event is a one-toggle wait that actually moves an edit at
the same radius and count, improves the anchor under the paired missing
label bound, and clears the original recording check and both CV/OLS4
gain checks. The controller cannot see any of those labels. It chooses
at most one wait per recording at the first eligible release query using
the same fixed sign and probability rules for all heads. A choice can
expire. The full-policy risk check uses the unchanged 2% rule. A separate
utility diagnostic replays the emitted slot identities at their original
times; it matches emitted counts and radii, but is not an online policy
and cannot certify the full policy's risk.

| Arm | Joint-event AP | Joint passes / 606 | Accepted / moved / expired | Gain vs anchor (pp, 95% CI) |
|---|---:|---:|---:|---:|
| Frozen anchor | not scored | 77 | 0 / 0 / 0 | reference |
| Self | 0.277 | 76 | 36 / 24 / 12 | +0.00001 [-0.00006, +0.00008] |
| Aligned peers | 0.263 | 76 | 37 / 29 / 8 | -0.0126 [-0.0379, +0.0009] |
| Wrong focal | 0.259 | 75 | 44 / 33 / 11 | -0.0013 [-0.0034, +0.00003] |

Aligned-minus-self AP is -0.0139 [-0.0500, +0.0088]; aligned-minus-wrong
is +0.0037 [-0.0061, +0.0150]. AP averages seeds with both classes within
each of five estimable locations; it is not directly comparable with the
global 9.1% event prevalence. Utility uses all seven locations. Intervals
resample locations 3,000 times after averaging seeds. Among 1,457
release instances, 132 are positive; two locations contain 1,395 releases
and 126 positives. The views repeat 202 recordings and are not independent
samples. There is no established benefit of this peer encoding, not a
claim that interaction never matters. The literature includes interaction
reasoning for forecasting, e.g. [EqMotion](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_CVPR_2023_paper.html);
the current experiment instead tests information for the deferral decision.

## Where the Full Policy Loses

The aligned-peer controller against its emission match gives -0.00005 pp
[-0.00143, +0.00124], not evidence of positive utility. Because at most one
slot changes per recording, a successful move and a dropped original edit
cannot occur in the same recording-seed view. Replaying the frozen heads
therefore decomposes the direct full-policy effect exactly: -0.00005 pp
from moved slots and -0.01257 pp from expiries. Of eight expiries, five
had no positive-prefix destination and three were blocked by available
credit, query capacity or an untouched edit. The worst is a single expiry
in location rdb2, seed 17 (-0.2604 pp of that location-seed's CV mass);
it had a legal radius candidate but none passed the positive-prefix rule.
The gain is not repaired by dropping this case from the analysis.

The experiment points to two gaps: few independent locations with useful
decisions, and the downside of losing a good immediate edit when no later
action qualifies. The next test should predict that downside separately
and compare a protected release decision with the original anchor at
the same offered budget, with additional development scenes if available.
The prior exact-count 138-versus-77 hindsight gap remains a diagnostic,
not a trained result. Nothing qualifies for deployment or independent
confirmation. No Stage5C, SMC, metric or seconds-level claim.

```sh
.venv-pytorch/bin/python scripts/run_m3w_wait_peer_value.py
.venv-pytorch/bin/python scripts/run_m3w_wait_peer_value.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_wait_peer_expiry.py
```

63 fresh native arm64 Torch fits with three seeds and 50,400 updates
completed in 49.45 seconds on CPU with four compute threads and no
DataLoader subprocesses. Checkpoints and PID heartbeat every 100 steps.
No new CREATE job. The independent DroneCrowd groups remain sealed.

[Registration](registration.json) | [Aggregate evidence](analysis.json) |
[Checkpoint receipts](head_freeze.json) | [Expiry audit](expiry/analysis.json).
