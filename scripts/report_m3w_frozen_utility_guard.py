"""Public-scalar report for the one-rule frozen utility/guard repair."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_m3w_frozen_utility_guard import summarize,api
HOME=ROOT/"outputs/publication_readiness_2026_09/european_frozen_utility_guard_v1"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def verified():
    cfg=json.loads((ROOT/"configs/m3w_frozen_utility_guard_v1.json").read_text())
    s=json.loads((HOME/"summary.json").read_text());c=json.loads((HOME/"complete.json").read_text());v=json.loads((HOME/"verification.json").read_text())
    if not v["all9_choice_matrices_exact"] or not v["all54_metric_groups_exact"] or v["summary_sha256"]!=sha(HOME/"summary.json") or v["complete_sha256"]!=sha(HOME/"complete.json"):
        raise ValueError("Separate decision/metric replay required")
    rows=[]
    for ref in c["groups"]:
        p=ROOT/ref["path"]
        if p.parent!=HOME/"groups" or sha(p)!=ref["sha256"]:raise ValueError("Public scalar identity changed")
        rows.append(json.loads(p.read_text()))
    if summarize(rows,cfg)!=s:raise ValueError("Public summary reconstruction differs")
    return s,rows,c,v,json.loads((HOME/"policy_freeze.json").read_text())


def render():
    s,rows,c,v,f=verified();ci=lambda x:f"{x['point']:+.5f} [{x['CI95'][0]:+.5f}, {x['CI95'][1]:+.5f}]"
    effects=[f"|{k}|{ci(a['known_pp'])}|{ci(a['completion_lower_pp'])}|" for k,a in s["effects"].items()]
    table=[]
    for arm in api.POLICIES:
        a=s["safety"][arm];switch=100*sum(r["policies"][arm]["switch_rate"] for r in rows)/len(rows)
        table.append(f"|{arm}|{ci(s['accuracy'][arm])}|{switch:.3f}%|{a['worst_known_easy_net_degradation_percent']:.3f}%|"
            f"{a['supported_record_views']}/{a['required_record_views']}|{a['known_all_risk_violations']}/{a['known_easy_risk_violations']}|{a['undefined_easy_views']}|")
    content=f"""# Frozen Utility / Variable Guard: Complete Bounded Repair

## Material Passport

Mode:run and numerical reproducibility validation. Fresh_run:one registered
decision repair,nine frozen five-policy action matrices and54metric groups,
plus separate choice/metric replay. Cached_verified:all27SOURCE-trained cost
critics and their outputs,original forecasts/inputs and exposed selection labels.
No new training,trajectory inference,cost inference or threshold search.

## Why This Was Tested

The preceding actual-seed disagreement experiment lost utility while reducing
some harm. Its shared input changed both ranking and guards. The causal-only
audit finds30,135repeated candidate views with old predicted utility>0,new real
guards passing,but new predicted utility<=0. These are predicted opportunities,
not oracle useful actions. This preregistered repair tests exactly one inference
factor rather than training another model or sweeping thresholds.

Keep the preceding disabled mean_absolute B-H utility/ranking fixed. Use real
or noise H/R and EH/ER guards at the unchanged2%. Choose the largest old positive
utility passing those guards,otherwise keep the same SOURCE floor. Guard benefit
is not used. The two learned quantities need not define a single coherent
posterior;recombination is not a calibration or safety theorem. All three
coupled controls must reproduce the preceding experiment exactly.

## Registered Contrasts

Primary:fixed-real minus fixed-noise paired completion lower. Advancement:
fixed-real minus coupled-disabled. Mechanism secondary:fixed-real minus coupled-real.

|Contrast|Known secondary,pp [95% CI]|Paired completion lower,pp [95% CI]|
|---|---:|---:|
{chr(10).join(effects)}

Primary pass:**{s['primary_pass']}**. Advancement pass:**{s['advancement_pass']}**.
Original full-record risk:**{s['original_record_risk_pass']}**.
Candidate:**{s['policy_candidate_pass']}**. Deployment changed:**false**.
Improving over a failed coupled policy alone does not meet advancement.

## Complete Policy And Risk Results

|Policy|Known ADE gain vs same floor,% [CI]|Mean intervention|Worst easy net|Supported repeated record views|Known all/easy risk violations|Undefined easy risk|
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

All38,102targets/28recordings/six exposed selection localities remain,including
848wholly unknown and15,820partially labelled outcomes. Original all/easy
positive-harm and easy-net2%budgets apply to every required recording. Benefits
do not cancel positive-harm numerators. Empty selected-easy reference is not
estimable,not safe zero.252views repeat28recordings across nine producer models.

Bootstrap3000localities after averaging SOURCEfolds/seeds within each of six
localities. Overlapping rows are not independent observations. Completion
bounds apply to wholly unknown rows under available-label ADE;they do not fill
missing steps of partial trajectories. Full-roster endpoint FDE is not identified.
No same-count comparison or joint-allocation contribution is claimed here.

## Execution And Reproduction

Decision freeze PID{f['pid']}:{f['seconds']:.6f}s,peak{f['peak_RSS_bytes']}B.
Evaluation PID{c['pid']}:{c['seconds']:.6f}s,peak{c['peak_RSS_bytes']}B.
Separate verification PID{v['pid']}:{v['seconds']:.6f}s. All nine complete action
matrices and54metric groups reproduce exactly. Cached critic costs are checked
by hash,not freshly inferred in this repair. The preceding27-model Torch replay
is a separate inherited receipt. Private arrays/weights remain outside Git.

```sh
.venv-pytorch/bin/python scripts/run_m3w_frozen_utility_guard.py register
.venv-pytorch/bin/python scripts/run_m3w_frozen_utility_guard.py freeze
.venv-pytorch/bin/python scripts/run_m3w_frozen_utility_guard.py eval
.venv-pytorch/bin/python scripts/run_m3w_frozen_utility_guard.py verify
.venv-pytorch/bin/python scripts/report_m3w_frozen_utility_guard.py
```

Completed endpoints are immutable;partial freeze/eval supports--resume. Existing
scalar reports and tests can be rerun without replacing completed phase receipts.
Local arm64CPU4/workers0,12hboundary/10GiBreserve;no CREATE submission.

## Scope And Limits

This is bounded model-selection development after exposed parent results,not
an untouched final test. SOURCE models,normalizers and priors remain frozen;
calibration/confirmation/DroneCrowd stay closed. Neither a positive secondary,
extra switching nor a replay pass constitutes deployment or a new world model.
No new risk-only head was trained,so this cannot establish that separate
supervision would succeed. The old coupled negatives and matched-joint failure
remain unchanged. Main PDF was not regenerated for this repair.

obs8/pred12/rawstride12,local-image pixel,detector-silver,available labels,
2.5D/pseudo-3D trajectory state. Not old rawt50,metric,seconds,human gold,true3D
or foundation. Stage5C/SMC remain off;submission readiness false.
"""
    (HOME/"readout.md").write_text(content)
    (HOME/"gates.md").write_text(f"""# Frozen Utility Repair Gates

- All nine causal action matrices frozen before this scoring:true.
- All coupled controls unchanged and54metric-group replay:true.
- Primary fixed-real minus fixed-noise paired-lower CI lower>0:{s['primary_pass']}.
- Advancement fixed-real minus old disabled paired-lower CI lower>0:{s['advancement_pass']}.
- Original full-record risk:{s['original_record_risk_pass']}.
- Candidate:{s['policy_candidate_pass']}.
- New training:false. New cost inference:false. Threshold search:false.
- Calibration/confirmation:not_run,closed. Deployment changed:false.
- Submission ready:false. Stage5C:false. SMC:false.
""")
    return dict(private_input_reads=0,summary_sha256=sha(HOME/"summary.json"),readout_sha256=sha(HOME/"readout.md"),
        primary_pass=s["primary_pass"],advancement_pass=s["advancement_pass"],original_record_risk_pass=s["original_record_risk_pass"],
        deployment_changed=False,submission_ready=False)


if __name__=="__main__":print(json.dumps(render()))
