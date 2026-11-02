"""Rebuild the selection readout from hash-bound public scalar evidence only."""
import hashlib
import json
from pathlib import Path

from src.evaluation import m3w_selection_forecast_readout as metrics

NAME = "european_selection_forecast_readout_v1"
RELATIVE = Path("outputs/publication_readiness_2026_09") / NAME


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(root):
    root = Path(root).resolve()
    home = root / RELATIVE
    read = lambda path: json.loads(path.read_text())
    summary, complete, verify, config = (
        read(home / "summary.json"), read(home / "complete.json"),
        read(home / "verification.json"), read(root / "configs" / (NAME.replace("european_", "m3w_") + ".json")),
    )
    identity = sha(home / "summary.json")
    if complete["summary_sha256"] != identity or verify["summary_sha256"] != identity:
        raise ValueError("Selection summary identity changed")
    if verify["complete_sha256"] != sha(home / "complete.json"):
        raise ValueError("Selection completion identity changed")
    if not verify["all28_raw_labels_reparsed"] or not verify["all54_metric_groups_exact"]:
        raise ValueError("Separate complete raw-label and metric replay required")
    rows = []
    for ref in complete["groups"]:
        path = (root / ref["path"]).resolve()
        if path.parent != (home / "groups").resolve() or path.suffix != ".json":
            raise ValueError("Only public scalar group evidence is admitted")
        if sha(path) != ref["sha256"]:
            raise ValueError("Group identity changed")
        rows.append(read(path))
    sites = sorted({row["site"] for row in rows})
    rebuilt = metrics.summarize(rows, sites, resamples=config["bootstrap_resamples"], seed=config["bootstrap_seed"])
    if any(summary[key] != value for key, value in rebuilt.items()):
        raise ValueError("Public scalar summary replay differs")
    if any(row["policies"]["history"]["full_roster_FDE_pixel"] is not None for row in rows):
        raise ValueError("Incomplete endpoints cannot be reported as full-roster FDE")
    return home, summary, complete, verify, rows, config


def render(root):
    home, s, complete, verify, rows, config = load(root)
    point = lambda x: f"{x['point']:+.2f} [{x['CI95'][0]:+.2f}, {x['CI95'][1]:+.2f}]"
    sites = sorted({row["site"] for row in rows})
    secondary = {}
    for other in ("current", "noise"):
        secondary[other] = metrics.interval(
            rows, lambda r, other=other: r["policies"]["history"]["known_ADE_gain_floor_percent"]
            - r["policies"][other]["known_ADE_gain_floor_percent"], expected_sites=sites,
            resamples=config["bootstrap_resamples"], seed=config["bootstrap_seed"],
        )
    seed_rows = []
    for seed in (17, 29, 43):
        selected = [r for r in rows if r["seed"] == seed]
        known = sum(r["paired_history_minus_disabled"]["known_pp"] for r in selected) / len(selected)
        lower = sum(r["paired_history_minus_disabled"]["completion_lower_pp"] for r in selected) / len(selected)
        seed_rows.append(f"|{seed}|{known:+.2f}|{lower:+.2f}|")
    forecast_rows = []
    for name in ("floor", "disabled", "current", "noise", "history", *(f"physical_{k}" for k in range(6))):
        risk = s["safety"][name]
        forecast_rows.append(f"|{name}|{point(s['accuracy'][name])}|{point(s['completion_lower'][name])}|"
                             f"{risk['worst_known_easy_net_degradation_percent']:.2f}%|"
                             f"{risk['supported_record_views']}/{risk['required_record_views']}|")
    locality_rows = [f"|{site}|{s['primary']['known_pp']['by_locality'][site]:+.2f}|"
                     f"{s['primary']['completion_lower_pp']['by_locality'][site]:+.2f}|" for site in sites]
    files = {
        "readout.md": f"""# First Frozen Selection-Locality Forecast Readout

## Scope And Provenance

Result source: **fresh_run** feature conversion, inference and selection evaluation;
**cached_verified** SOURCE-trained checkpoints and SOURCE-only normalizers.
No new training, threshold search, learned preprocessing fit or model selection
from these outcome values was performed in this readout. It is a supporting
forecast/input ablation, not the paper's joint-versus-independent policy result.

The 36 real Torch models cover three actual forecaster training seeds (17,29,43),
three SOURCE producer folds and four detector representations. All predictions
were frozen and fully replayed before the six model-selection localities' labels
were opened. These six localities are **now exposed for model selection**, not an
untouched test. Risk calibration, final confirmation and DroneCrowd remain closed.

## Fixed Population

All 38,102 past-supported targets from 28 recordings remain. There are 37,254
nonempty-label targets, 21,434 complete-12 targets, 15,820 partially labelled
targets, 848 wholly unknown targets and 27,694 available final endpoints.
No future-completeness filter selected the cohort. The task is obs8/pred12 at
raw-frame stride12 in local-image pixels. Detector-derived silver is not human gold.

ADE is averaged over each target's **available future labels**, not complete12 ADE.
Full-roster endpoint FDE is not estimable; available-endpoint FDE is diagnostic.
The registered completion convention bounds wholly unknown target costs using
forecast disagreement; it does not recover every unobserved step of partial
tracks or turn this into a fully observed 12-step benchmark.

## Registered Primary And Secondary

|History minus disabled, percentage points|Point [95% locality bootstrap CI]|
|---|---:|
|Registered paired completion-lower primary|{point(s['primary']['completion_lower_pp'])}|
|Known-label secondary|{point(s['primary']['known_pp'])}|
|Paired completion-upper diagnostic|{point(s['primary']['completion_upper_pp'])}|

The known secondary is positive in all six localities, but the registered lower
primary is negative in all six. **The primary fails.** Missing-label uncertainty
is not observed hidden harm. Independent known risk violations also prevent safety.
Individual lower-bound gains must not be subtracted to replace the paired lower.

Bootstrap: 3,000 resamples of the six localities after averaging SOURCE folds
and training seeds. Repeated predictions are not independent targets. These are
conditional model-selection development intervals, not final confirmation.

|Representation|Known-label ADE gain vs SOURCE floor, % [CI]|Individual completion-lower gain, % [CI]|Worst known easy net degradation|Risk-supported repeated record views|
|---|---:|---:|---:|---:|
{chr(10).join(forecast_rows)}

Physical0=constant position; physical1=causal CV; physical2=damped .90;
physical3=damped .97; physical4=OLS4; physical5=OLS8. The reference is the
**producer's SOURCE-selected fixed baseline**, not a validation-selected or
per-row oracle baseline. It differs by SOURCE fold. Every repetition retains
the same reference within its matched representation comparison.

The detector branch has 512 parameters within 89,026 total forecaster parameters.
Disabled uses zero detector contribution, so it is not an active-capacity-matched
control. Current and noise activate that same branch. Exploratory known-label
history-minus-current: {point(secondary['current'])} pp; history-minus-noise:
{point(secondary['noise'])} pp. These additional comparisons do not change the
registered primary and are not multiplicity-adjusted new primary claims.

## Localities And Actual Training Seeds

|Selection locality|Known paired effect, pp|Paired completion lower, pp|
|---|---:|---:|
{chr(10).join(locality_rows)}

|Actual forecaster training seed|Mean known paired effect, pp|Mean paired lower, pp|
|---|---:|---:|
{chr(10).join(seed_rows)}

These are paired representations of actual training seeds, not three stochastic
resamples of a single trained forecaster. Each seed is still repeated across
the same six localities and three SOURCE folds.

## Safety And Decision

History full substitution supports **0/252** repeated recording-risk views:
249 known all-risk and 247 known easy positive-harm violations remain. Worst
known easy net degradation is **47.98%**. The original positive-harm and easy-net
budgets remain 2%; favorable average ADE cannot offset positive harm.

The floor itself has no intervention and zero known harm. Its empty selected
easy denominator is not estimable, not proof of a safe positive learned switch.
The 252 views are 28 recordings repeated for nine producers, not 252 recordings.

**No new deployment. No submission-readiness claim.** There is known-label
forecast-input utility on previously unscored localities, but no supported safe
complete policy, joint-decision contribution or independent confirmation yet.
Current evidence remains 2.5D/pseudo-3D agent-trajectory state modeling, not true3D,
metric/seconds-level prediction or a foundation model. Stage5C and SMC stay off.

## Execution And Replay

Readout PID {complete['pid']}: {complete['seconds']:.6f}s, peak RSS
{complete['peak_RSS_bytes']:,}B. Separate raw-label/metric replay PID {verify['pid']}:
{verify['seconds']:.6f}s, peak {verify['peak_RSS_bytes']:,}B; all28 raw labels and
all54 metric groups reproduce exactly. Full36 model inference had separately
replayed exactly before outcome opening. That inference replay is not repeated
inside the label/metric verification.
""",
        "failure_analysis.md": """# Failure Analysis And Next Experiment

1. **Generalization of known-label input utility: supported secondary.** Historical
   detector context improves paired known ADE on every selection locality. Current
   and noise controls do not explain away the point gain. This is not automatically
   a JEPA, dynamics, joint-policy or deployable contribution.
2. **Missing-outcome identification: primary fails.** The all-target paired lower
   is negative, while known-label utility is positive. Do not delete 848 unknown
   or 15,820 partial targets or replace the primary with a complete-only subset.
3. **Full-substitution safety: observed failure.** Easy degradation reaches47.98%
   and known positive-harm violations occur independently of missing outcomes.
   This is not merely lack of statistical support or bootstrap conservatism.
4. **Intervention control: not evaluated here.** A better candidate forecast is
   not a learned decision about whether to use it. There is no joint-controller
   result in this supporting experiment. Full substitution cannot replace the
   required same-forecaster independent-versus-joint allocation comparison.
5. **Capacity and label caveats.** Disabled has an unused detector branch;
   current/noise are necessary capacity controls. Detector-silver partial ADE is
   not a human-gold full-trajectory benchmark. Only six selection localities
   determine the cluster interval; no final role is exposed.

Next action: preserve this fixed candidate bank; build a matching SOURCE-only
cost/controller interface using the actual partial-neighbor geometry. Train
unsafe-admission/false-rejection prediction only on SOURCE fitting targets;
use selection outcomes for model selection, never fitting weights or claiming
untouched confirmation. Compare independent, scene-uniform and joint decisions
with the same forecast bank and original 2% full-policy rules. Freeze the policy
before independent risk calibration and final confirmation. Do not run another
test-threshold sweep or treat stronger average forecasts as safe deployment.
""",
        "reproduction_zh.md": """# 复现与数据边界

本次不是新训练。36个已训练的真实Torch模型来自3个实际训练种子、3个SOURCE折和4种输入。
本轮重新构建38,102行输入，重新推理并完整重放36组预测，随后才读取6个模型选择场景的标签。
54组指标和28个录像的标签已单独从原始数据复核。校准集、最终确认集和DroneCrowd未读取。

公共标量报告可运行：

```bash
.venv-pytorch/bin/python scripts/report_m3w_selection_forecast.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_selection_forecast_readout.py tests/test_m3w_selection_forecast_report.py
```

报告重建只读取仓库中的标量JSON，不读取私有轨迹、特征、预测、标签或checkpoint。
它核对54组文件哈希、完成回执和复核回执，重算6个场景为单位的3000次bootstrap。
完整原始数据复现需要依法获取原始数据及私有SOURCE checkpoint，不能把公共标量重建说成重训。

完整流程入口为run_m3w_selection_forecast.py和run_m3w_selection_forecast_readout.py。
已完成阶段拒绝覆盖原结果，不应删除回执后重跑。中断阶段使用原绑定和--resume。
CPU4线程、单进程、arm64 .venv-pytorch；checkpoint/heartbeat和原始运行PID均有记录。

本次主要结果：已知标签增益为正，但预先登记的保守主指标为负，easy和正伤害风险不通过。
因此不部署、不宣称投稿就绪；它是模型选择证据，不是最终独立确认。
obs8/pred12、raw stride12、local-image pixel、detector-silver；不是米、秒、true3D或foundation。
Stage5C不执行，SMC不启用。
""",
    }
    for name, content in files.items():
        (home / name).write_text(content)
    receipt = dict(result_source="cached_verified_public_scalar_rebuild_of_fresh_readout",
                   summary_sha256=sha(home / "summary.json"), scalar_groups=len(rows),
                   public_scalar_summary_exact=True, private_input_reads=0,
                   exploratory_history_controls=secondary,
                   files={name: sha(home / name) for name in files},
                   primary_pass=s["primary_lower_positive"],
                   original_record_risk_pass=s["history_original_record_risk_pass"],
                   new_training=False, deployment_changed=False, submission_ready=False)
    (home / "report_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
