# Verification And Remaining Gaps

| Operation | PID | Exit | Seconds | Peak RSS bytes |
|---|---:|---:|---:|---:|
| Boundary/cache audit | 44790 | 0 | 113.498 | 7666286592 |
| Full primary model pilot and repeat | 45301 | 0 | 2.966 | 625999872 |
| All new fits and cached controls | 45347 | 0 | 8.053 | 811171840 |
| Complete model refits | 45404 | 0 | 12.275 | 859308032 |
| Fit-forbidden prediction replay | 45651 | 0 | 2.195 | 759021568 |

The short fitting runtime follows verified feature-cache reuse, not a quick
sample substitution. The full planned folds, rows and tree counts are retained.
Actual forest configuration uses four threads; the inherited runtime banner
still prints one for its old forest field. Torch CPU4/inter-op1/workers0 is
unchanged. No new CREATE job, background training or resource downscope.

All 16 new model objects and eight controls reproduce by complete refit. All
original target RMS values match, and each shared32 control is exactly the
shared64 first32-tree prefix. Replay forbids both fitting entry points and
reconstructs all 24 saved predictions and metrics. All 38 public files present
when replay started remain unchanged; this report was added afterward.

Twenty-seven focused tests passed in3.60s across channel training/reporting,
boundary attribution, original decision semantics, cost partitions and peer
probes. These include source-held label perturbation, cross-channel target
independence and undefined-support preservation. The full historical test suite
was not rerun. Tests and exact refits are engineering evidence, not a pass of
the scientific information screen (0/3) or original safety conditions.

Analysis SHA256:
`cd663b5f89e44f8de0af0ccc7960381ad92804606012245f86796673b9442d9f`

Compiled manuscript SHA256 remains:
`d9babe3ddbac858bcc1ac30e3ba98b332bed625c78ff9231d1d42a3e30b11772`

The separate paper addendum is not incorporated into that PDF. The paper still
lacks a successful main-method result under the original full-view contract
and independent confirmation. Original24 primary remains not estimable.
No new nested-C/H readout, reserved-data use, identity exemption, deployment
upgrade, metric/time claim, Stage5C or SMC. Submission readiness remains false.
