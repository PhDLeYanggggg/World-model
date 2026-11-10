# Songdo: which recipients account for the joint contrast?

The original Songdo test kept two edits at the same peak radius, but
the arms could edit different vehicles. I decomposed the direct trajectory
contrast into vehicles edited by joint alone, by the control alone,
and by both arms. All terms use the original whole-flight CV denominator.
Neither the forecasts nor the release rules were changed.

| Joint minus | Different recipients, combined (pp) | Shared recipients (pp) | Total conservative effect (pp) |
| --- | ---: | ---: | ---: |
| Independent | -0.00044684 | +0.00000465 | -0.00044218 |
| Additive | -0.00045268 | +0.00000510 | -0.00044758 |
| Same family | +0.00014172 | -0.00000085 | +0.00014087 |

In the comparison with independent selection, the two exclusive terms
are -0.00011661 pp for joint-only vehicles and -0.00033023 pp for
independent-only vehicles. The combined exclusive-recipient contribution
is negative on the observed grid (-0.00027669 pp) before the missing
waypoint debit. The shared-recipient contribution is small and has no
missing-waypoint debit. In 25 of 480 views the arms choose the same two
vehicles; only 18 shared recipient paths differ across the whole cohort.

The 95% source bootstrap for the independent exclusive-recipient term
is [-0.00150967, +0.00023373] pp. The total interval remains
[-0.00149870, +0.00023659]. The source-weighted arithmetic attribution
locates the gap in the fixed forecasts; it does not establish a
population-wide effect. The advantage over the restricted same-family
control likewise comes from different recipients, not improved paths
for the shared recipients. This does not isolate the causal effect
of forcing the policies to edit the same vehicle set.

Unlike joint forecast metrics that measure compatibility across trajectories
([Weng et al.](https://openaccess.thecvf.com/content/ICCV2023/html/Weng_Joint_Metrics_Matter_A_Better_Standard_for_Trajectory_Forecasting_ICCV_2023_paper.html)),
the partition here is exact only for the additive trajectory score.
The separate relative-motion results stay unchanged.

The audit reread only the 160 previously scored confirmation flights and
reproduced the observed and conservative trajectory contrasts in all
480 flight-seed views, including the 14 linked source component means
and their published intervals. No calibration or new cohort outcomes
were opened, and no fit, threshold or deployment decision changed.
This is retrospective attribution, not another independent result.

[Protocol](protocol.md) | [Readout](readout.json) |
[v192 manuscript](../evidence_manuscript_v192_songdo_attribution/README.md)

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_songdo_recipient_attribution.py
```

The complete local run took 54 seconds. It saves one scalar receipt
per flight and reuses completed receipts on resume.
