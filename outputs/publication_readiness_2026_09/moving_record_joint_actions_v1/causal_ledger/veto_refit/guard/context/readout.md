# Closer First-Query Motion Does Not Improve Calibration Transfer

I kept each parent model and its ten fitting, three calibration
and one evaluation source fixed. The context rule selects one
recording per calibration source using the **first current query**
of the evaluated recording. Its eight observed steps supply all
features. The blind and farthest controls use the same three-recording
budget and the same threshold rule. No model was refitted. Each
policy has 6,344 switches at exactly the same queries.

| Threshold Selection | Direct Lower vs Base (pp) | Selected Reopened Actions | Easy Net Above 2% | Easy Positive Harm Above 2% | Undefined Easy | Full Passes / 102 |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0 | 0 | 2 | 59 | 41 | 2 |
| Unguarded reopening | +0.20260 | 2,670 | 8 | 62 | 37 | 3 |
| Pooled radius guard | +0.08172 | 464 | 4 | 64 | 35 | 3 |
| Context-blind, three recordings | +0.17014 | 527 | 5 | 61 | 38 | 3 |
| Nearest, three recordings | +0.16274 | 565 | 5 | 62 | 37 | 3 |
| Farthest, three recordings | +0.19028 | 1,188 | 3 | 61 | 38 | 3 |

The primary nearest-minus-blind direct completion lower is
**-0.00905 pp [-0.01875, -0.00111]**. Nearest minus farthest is
-0.03392 pp [-0.06176, -0.01215]; nearest minus pooled is
+0.07986 pp [-0.00955, +0.20952]. These are 3,000 paired draws
of 14 sources after averaging three actual forecaster seeds.
The farthest control reopens over twice as many selected actions
as nearest, so this is not a fixed reopening-count comparison
or a reason to promote the farthest rule.

Nearest lowers mean standardized profile distance from 2.372 to
2.081, changes the calibration recordings on 63/102 views and
the threshold on 40/102, and changes 771 row decisions relative
to blind. It fails the inner preservation conditions on four
outer views versus five for blind; the full 2% passes are
unchanged. Hard known gain vs base is +0.22316 pp
[0.01066, 0.48895] for nearest and +0.23160 pp
[0.01757, 0.49859] for blind. Neither establishes a hard
gain specifically from matching. This is not a null exposure
test: matching changed what the policy saw and selected.

The experiment shows that **these first-query features** are not
a useful transfer key for this rule and donor pool. It does not
rule out better context, larger calibration sets, or conditional
shift methods. Unlike
[weighted conformal prediction under covariate shift](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html),
this heuristic does not estimate a source-to-target likelihood ratio
or establish invariance of the outcome distribution given features.
There is no formal coverage guarantee.

**Decision.** Do not replace the pooled reference with nearest or
farthest matching. Test calibration sample size and actual
release-risk relevance before changing the profile again. The
development gain from reopening remains real, but all variants
miss the original full-risk requirement and independent confirmation
is still absent. Original 2% rule, support, deployment, Stage5C and SMC
are unchanged. All trajectories remain image-coordinate 8-to-12
annotation-step evidence.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_context_calibration.py freeze
.venv-pytorch/bin/python scripts/run_m3w_context_calibration.py score
.venv-pytorch/bin/python scripts/run_m3w_context_calibration.py replay
.venv-pytorch/bin/python scripts/audit_m3w_context_calibration_transfer.py
```

Design `6a92a6ce` precedes the 68.66 s native arm64 decision
run. The three cached controls and complete score replay are
exact; 60 scoped tests pass. [Protocol](protocol.md) |
[Replay receipt](replay_execution.json) |
[Transfer diagnostic](transfer_diagnostic.json)
