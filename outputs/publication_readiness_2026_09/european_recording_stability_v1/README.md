# Recording-Deletion Stability Control

Status: all72 groups computed and exactly replayed; the stability-filter repair
failed. Primary complete support drops23->8, with no same-count utility lift.
[Conclusions](conclusions.md), [failure analysis](failure_analysis.md),
[summary](summary.json) and [next action](next_action.md).
This is source-only diagnosis, not a promoted model or independent confirmation.

The [protocol](protocol.md) fixes both calibration arms and the entire72-head
roster. Parent leave-one-recording-out actions must replay before any deletion
diagnostic. Held-recording outcomes are excluded from every nested calibrator.
Zero support means unestimable, not safe.

## Run and Resume

From the repository root, using the existing native arm64 environment:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m pytest -p no:cacheprovider tests/test_m3w_recording_stability.py -q
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_recording_stability.py pilot
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_recording_stability.py run
```

If an execution is verified interrupted, use the last command with `--resume`.
Existing aggregates are reproduced exactly, never silently replaced. A completed
run must not be launched again. Logs expose the PID, current group, elapsed time
and peak memory. The private heartbeat is at
`data/stage_cvpr2027_experiments/european_recording_stability_v1/heartbeat.json`.

All source inputs are streamed from the already completed CREATE experiment.
No new HPC job, login-node fitting, local raw/array cache, or simulation changes.
Small group summaries are the checkpoints; all nested computation has exact
replay. Frozen code/config/protocol hashes are in `registration.json`.

The same-count comparator is an exact expected utility under uniform thinning
within a recording. It is not same-frame random selection, a realized policy,
an expected ratio, or a safety certificate. Independent roles remain closed,
the original2%risk budget remains, and Stage5C/SMC stay off.

## Completed-Run Verification

Do not repeat `run` on the completed directory. Verify the frozen outputs without
training, array access or Torch import:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/verify_m3w_recording_stability.py
```

This checks group hashes, prior anchors, partition arithmetic and the nominal
locality-bootstrap results. It does not independently reimplement nested fitting.
The [receipt](verification.json) reports1802 aggregate checks and3371 anchor
fields. Computation completed in162.20s on native arm64 with0.924GB peak RSS;
no CREATE job or new neural training was required.

## 中文操作说明

本轮复用了 CREATE 上已校验的72组输入，在本地内存计算，没有复制大缓存。
72组全部完成并逐组重放，结果不是只跑试验组。11项方法检查和5项汇总检查通过，
不等于科研假设成立，也不等于全仓库测试全部通过。

结果是负面的：主对照从23组完整通过降到8组；删除的介入全部来自校准支持不足，
没有识别出原有的超标风险。不要据此改部署或打开独立测试数据。查看结论和失败
分析后，使用上方只读核验命令检查产物；不要向已经完成的目录重复提交或恢复。
