# Train-Scale Readout Conditioning

## Conclusion

Neither conditioned arm beats stationary CV. No new deployment.

Twenty-four fresh Torch heads complete 240,000 updates. The 24 unconditioned
importance-corrected heads are cached_verified controls. Registration was
committed as 5aa189b1 before the included 100-update pilot. No held-driven
checkpoint, seed, threshold or early-stop selection.

## What Changed

The fixed training-gradient audit found severe output-layer concentration but
no large clipping-induced mean-direction reversal at the final checkpoints.
This trial changes only q to q/c before the existing radial bound. Each c is
the median training radius divided by the existing training-loss scale, floored
at one. It does not change the represented forecast class or per-row bounds.
It does change parameter conditioning and optimizer geometry, including effective
step sizes and weight decay. It is not a pure clipping ablation.

All 63,960 parameters, inputs, training complements, original loss, sample streams,
seeds, 10,000-update budgets, schedule and cap5 are retained. Frozen ResNet18
features remain frozen; these are trajectory readout heads, not end-to-end
visual world-model training.

## Primary Results

| Arm | Source | Equal-site gain (%) | Conditional 95% interval | Static harm (px) | Positive held fits |
| --- | --- | ---: | --- | ---: | ---: |
| geometry_control | cached_verified_recomputed | -0.03206 | [-0.05775, -0.01284] | 0.0006392 | 0/12 |
| centered_control | cached_verified_recomputed | -0.27459 | [-0.68041, -0.03179] | 0.0078266 | 0/12 |
| geometry_conditioned | fresh_run | -0.00025 | [-0.00063, -0.00003] | 0.0000064 | 0/12 |
| centered_conditioned | fresh_run | -0.00134 | [-0.00366, -0.00005] | 0.0000412 | 0/12 |

| Fixed contrast | Gain difference (pp) | Conditional interval |
| --- | ---: | --- |
| geometry_conditioned minus geometry_control | +0.03181 | [+0.01282, +0.05720] |
| centered_conditioned minus centered_control | +0.27325 | [+0.03174, +0.67675] |
| centered_conditioned minus geometry_conditioned | -0.00109 | [-0.00302, -0.00002] |

The primary is the ratio of equal-site mean past-normalized ADEs. Seed errors
are averaged, not prediction-ensembled. The 2,000 bootstrap draws are conditional
on four explored source sites and shared training folds. They are not independent
confirmation or a guarantee of generalization. No main, outer or bookstore scores.

## Every Site and Seed

| Arm | Excluded source site | Held gain (%) | Hard gain (%) | Seed gains (%) | Training seed gains (%) |
| --- | --- | ---: | ---: | --- | --- |
| geometry_control | coupa | -0.01495 | -0.00050 | -0.01604, -0.01244, -0.01636 | -0.01459, -0.01013, -0.01460 |
| geometry_control | deathCircle | -0.01087 | -0.00035 | -0.01100, -0.00838, -0.01322 | -0.01231, -0.01032, -0.01717 |
| geometry_control | gates | -0.03867 | -0.00056 | -0.03477, -0.03519, -0.04605 | -0.01144, -0.01215, -0.01644 |
| geometry_control | hyang | -0.07012 | -0.00363 | -0.12672, -0.03753, -0.04611 | -0.01759, -0.01567, -0.02167 |
| centered_control | coupa | -0.02330 | -0.00061 | -0.02651, -0.02125, -0.02215 | -0.01535, -0.01533, -0.01407 |
| centered_control | deathCircle | -0.03976 | +0.00007 | -0.05790, -0.02408, -0.03732 | -0.02265, -0.02050, -0.01871 |
| centered_control | gates | -0.11361 | +0.00251 | -0.10659, -0.08108, -0.15317 | -0.01454, -0.01696, -0.02053 |
| centered_control | hyang | -0.91425 | +0.00167 | -0.96623, -0.73121, -1.04531 | +0.00474, +0.00803, +0.01254 |
| geometry_conditioned | coupa | -0.00003 | -0.00000 | -0.00003, -0.00004, -0.00004 | -0.00003, -0.00003, -0.00003 |
| geometry_conditioned | deathCircle | -0.00002 | -0.00000 | -0.00002, -0.00002, -0.00002 | -0.00003, -0.00003, -0.00002 |
| geometry_conditioned | gates | -0.00010 | -0.00000 | -0.00011, -0.00009, -0.00009 | -0.00004, -0.00003, -0.00003 |
| geometry_conditioned | hyang | -0.00084 | +0.00001 | -0.00237, -0.00007, -0.00009 | +0.00011, -0.00004, -0.00004 |
| centered_conditioned | coupa | -0.00004 | -0.00000 | -0.00005, -0.00004, -0.00004 | -0.00003, -0.00003, -0.00003 |
| centered_conditioned | deathCircle | -0.00005 | +0.00000 | -0.00006, -0.00004, -0.00006 | -0.00003, -0.00003, -0.00004 |
| centered_conditioned | gates | -0.00018 | +0.00000 | -0.00019, -0.00014, -0.00022 | -0.00003, -0.00004, -0.00003 |
| centered_conditioned | hyang | -0.00494 | +0.00008 | -0.00526, -0.00257, -0.00699 | +0.00038, +0.00014, +0.00069 |

## Error Magnitudes and Candidate Utility

| Arm | Native ADE | Native FDE | Nonzero-target gain (%) | Binary oracle (%) |
| --- | ---: | ---: | ---: | ---: |
| geometry_control | 1.1280880 | 2.3629168 | -0.00715 | 0.002304 |
| centered_control | 1.1322034 | 2.3826933 | -0.01826 | 0.064806 |
| geometry_conditioned | 1.1276568 | 2.3618742 | -0.00005 | 0.000030 |
| centered_conditioned | 1.1276762 | 2.3619957 | -0.00006 | 0.000339 |

Native values are annotation pixels. Easy percentage degradation is undefined
on these zero-error static-CV targets, not a 2% gate pass. Binary oracles use
future labels only for diagnosis, not as inference inputs or learned results.

## Optimization and Verification

Logged gradient clipping fractions range 0.0000 to 0.0000,
versus 1.0 in all predecessor fits. Logs cover update1/every100, not every update.
This quantifies changed optimization behavior, not prediction success.

All 24 train/held forecasts replay exactly; draw streams regenerate and match
controls. Six OOF archives recompute. Thirty-two future-label poison queries and
24 forbidden training-role checks pass. Completed resume adds zero updates and
preserves 82 artifacts. Thirty-three scoped tests pass;
the full legacy suite was not rerun. CPUarm64, four compute threads, one inter-op,
zero workers. Summed fitting time 676.292s including the pilot.

Fresh: readout fits, analysis and verification. Cached_verified: source data,
fixed embeddings, event mapping and previous control fits. Not_run: independent
confirmation, main/outer/external readout or a deployable intervention policy.
Offline supplied annotations may contain later interpolation controls; not
strict sensor-as-of. Eight observed/twelve predicted steps, stride12rawframes;
not metric or seconds. No true-3D/foundation claim, Stage5C or SMC.

See [fixed design](../source_conditioned_readout_decision.md), [analysis](analysis.json),
[verification](verification.json), [failure analysis](failure_analysis.md),
[gates](gates.md) and [reproduction](reproducibility.md).
