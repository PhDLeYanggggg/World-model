# Source-Held Error Discovery: Completed

I compared errors discovered on an unseen fitting source against errors
on an already seen source using equal-sized discovery networks. Both
student arms start from the same frozen kinematic neural model. The only
changed training ingredient is which observed error channels receive
weight10. Architecture, sampling, optimizer budget and 2% rules are fixed.

Eight randomly initialized discovery heads and six matched student heads
completed 28,000 primary updates. Independent full teacher/student refits
add 4,000 updates; the real resume pilot adds200. Predictions, decisions
and metrics reproduce exactly. This is real neural risk-head training,
not new trajectory dynamics, a new JEPA fit or independent confirmation.

## Held Development Results

| Model | Known ADE gain over floor % | Hard gain % | Intervention % | Full original risk/support |
|---|---:|---:|---:|---:|
| readout_veto | 4.3233 | 5.3547 | 22.0115 | 0/17 |
| kinematic_veto | 3.9418 | 4.8582 | 17.3155 | 0/17 |
| continued_mse_veto | 3.6006 | 4.6257 | 15.0144 | 0/17 |
| false_safe_priority_veto | 3.0323 | 3.8505 | 13.5856 | 0/17 |
| in_source_errors_veto | 3.2049 | 4.0374 | 13.9823 | 0/17 |
| held_source_errors_veto | 2.9544 | 4.0590 | 14.1132 | 0/17 |

Held-error minus in-source-error known contrast: **-0.2505 pp**.
Missing-outcome completion bounds: [-0.8846, 0.3837] pp.
These bounds are not sampling confidence intervals.
Prospective expansion screen: **False**. Original 24-group primary remains NA.
No model is deployed or certified by this report. No identity certificate
or changed support requirement. Empty matched controls remain NA.

## Safety

| Student | Known all violations | Known easy violations | Undefined selected easy | Worst defined easy-net % |
|---|---:|---:|---:|---:|
| in_source_errors | 14 | 14 | 0 | 24.9970 |
| held_source_errors | 14 | 14 | 2 | 64.1321 |

## Runtime

| Phase | PID | Seconds | Peak RSS bytes |
|---|---:|---:|---:|
| pilot | 80481 | 29.22 | 3196977152 |
| teachers | 80596 | 726.52 | 9170108416 |
| students | 81750 | 534.17 | 7544979456 |
| evaluate | 82614 | 46.57 | 4474388480 |
| verify | 82755 | 44.91 | 4453957632 |

Native arm64 Torch CPU4, interop1, workers0; no resource probing.
The first pilot failed before training because the immutable cache
rejects reduced role populations. The repaired interface subsets whole
sources/queries after reading the original cache. Both registrations
and the preflight amendment remain; the evaluation contract is unchanged.

## Limits And Provenance

New teachers, students and evaluation: fresh_run. Original forecasts,
router, warm heads and old control actions: cached_verified by hashes,
schemas and source exclusion. Old control metrics are freshly recomputed
and exactly matched. No new H/independent test: not_run, reserved.
B contains only two/three/three source groups. Individual teachers see
one/two/two sources; row counts and difficulty differ between teachers.
One producer seed and seven exposed held SOURCE units cannot establish
independent generalization. No favorable-context selection is allowed.
Native obs8/pred12 image pixels, not verified metric or seconds; not
true3D, not a foundation world model. Stage5C and SMC remain disabled.

[Protocol](protocol.md) | [Failure analysis](failure_analysis.md) | [Exact replay](verification.json)
