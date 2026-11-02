"""One frozen-critic decision repair,not a new model or threshold search."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time

if platform.system()=="Darwin" and platform.machine()!="arm64":raise RuntimeError("Native arm64 before Torch")
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):os.environ.setdefault(key,"4")

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import run_m3w_selection_uncertainty as parent
from src.world_model import m3w_frozen_utility_guard as api

prior=parent.prior
PUBLIC=ROOT/"outputs/publication_readiness_2026_09/european_frozen_utility_guard_v1"
PRIVATE=ROOT/"data/stage_cvpr2027_experiments/european_frozen_utility_guard_v1"
CONFIG=ROOT/"configs/m3w_frozen_utility_guard_v1.json";REG=PUBLIC/"registration.json"
sha,artifact,checked,write,once=prior.sha,prior.artifact,prior.checked,prior.write,prior.once
FILES=(Path(__file__),CONFIG,ROOT/"src/world_model/m3w_frozen_utility_guard.py",
       ROOT/"tests/test_m3w_frozen_utility_guard.py",PUBLIC/"protocol.md")


def admission(register=False):
    cfg=json.loads(CONFIG.read_text());_,rows,ids=parent.admission()
    v=json.loads((parent.PUBLIC/"verification.json").read_text())
    if not v["all27_new_inferences_exact"] or not v["all54_metric_groups_exact"] or v["summary_sha256"]!=sha(parent.PUBLIC/"summary.json"):
        raise ValueError("Verified complete parent comparison required")
    if cfg["positive_harm_budget"]!=.02 or cfg["easy_net_degradation_budget"]!=.02 or any(cfg[k] for k in
        ("new_training","threshold_search","calibration_access","confirmation_access","stage5c_executed","smc_enabled")):
        raise ValueError("Frozen-model single-rule test with original2%risk required")
    value=dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        parent_registration=artifact(parent.REG),parent_policy_freeze=artifact(parent.PUBLIC/"policy_freeze.json"),
        parent_summary=artifact(parent.PUBLIC/"summary.json"),parent_verification=artifact(parent.PUBLIC/"verification.json"),
        policies=list(api.POLICIES),primary=cfg["primary"],advancement=cfg["advancement"],
        selection_already_exposed=True,new_training=False,threshold_search=False,new_joint_solver=False,
        calibration_access=False,confirmation_access=False,stage5c_executed=False,smc_enabled=False)
    if register:once(REG,value)
    if json.loads(REG.read_text())!=value:raise ValueError("Decision repair changed")
    if not register:
        for p in (*FILES,REG):
            blob=subprocess.run(["git","show","HEAD:"+str(p.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
            if hashlib.sha256(blob).hexdigest()!=sha(p):raise ValueError("Commit repair before execution")
    return cfg,rows,ids


def frozen_case(row):
    data,paths,causal=prior.selection_bank(row);predictions={};saved={};refs={}
    for arm in ("disabled","real","noise"):
        ref=json.loads((parent.PUBLIC/"policies"/(row["trial"]+"_"+arm+".json")).read_text())
        predictions[arm]=np.load(checked(ref["prediction"]),mmap_mode="r",allow_pickle=False)
        saved[arm]=np.load(checked(ref["choices"]),allow_pickle=False);refs[arm]=artifact(parent.PUBLIC/"policies"/(row["trial"]+"_"+arm+".json"))
    choices=api.policies(predictions,causal["allowed"],saved)
    return data,paths,choices,api.causal_diagnosis(predictions,causal["allowed"],choices),refs


def runtime(phase):
    parent.torch.set_num_threads(4);parent.torch.set_num_interop_threads(1);start=time.monotonic()
    def interrupt(*_):raise KeyboardInterrupt("Resume completed decision cases")
    signal.signal(signal.SIGTERM,interrupt)
    def guard():
        if shutil.disk_usage(ROOT).free<10*2**30:raise OSError("Preserve10GiBreserve")
        if time.monotonic()-start>43200:raise TimeoutError("12hboundary;resume")
    def beat(**kw):
        guard();r=dict(pid=os.getpid(),utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),phase=phase,**kw)
        write(PRIVATE/"heartbeat.json",r);print(json.dumps(r),flush=True)
    def receipt():return dict(pid=os.getpid(),seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,registration_sha256=sha(REG))
    beat(state="runtime_ready",new_training=False,new_cost_inference=False,threads=4,workers=0)
    return beat,guard,receipt


def freeze(resume=False):
    cfg,rows,_=admission();beat,guard,receipt=runtime("decision_freeze")
    if (PUBLIC/"policy_freeze.json").exists():raise ValueError("Complete decision freeze immutable")
    refs=[];diagnoses=[]
    for row in rows:
        guard();p=PUBLIC/"policies"/(row["trial"]+".json")
        if resume and p.exists():
            v=json.loads(p.read_text());checked(v["choices"])
            if v["registration_sha256"]!=sha(REG):raise ValueError("Policy checkpoint differs")
            refs.append(artifact(p));diagnoses.append(v["causal_diagnosis"]);continue
        data,_,choices,d,parents=frozen_case(row);home=PRIVATE/"choices";home.mkdir(exist_ok=True)
        target=home/(row["trial"]+".npy");prior.frozen.inputs.store_array(target,np.stack([choices[k] for k in api.POLICIES]))
        once(p,dict(trial=row["trial"],choices=artifact(target),policy_names=list(api.POLICIES),parents=parents,
            rows=len(data["ids"]),causal_diagnosis=d,registration_sha256=sha(REG),future_labels_read=False))
        refs.append(artifact(p));diagnoses.append(d);beat(state="decision_frozen",trial=row["trial"],cases=len(refs),total=9)
    once(PUBLIC/"policy_freeze.json",dict(**receipt(),policies=refs,labels_read_for_this_comparison=False,
        new_training=False,new_cost_inference=False,threshold_search=False,
        causal_diagnosis=dict(row_views=sum(d["row_views"] for d in diagnoses),
            original_positive_real_guard_action_views=sum(d["original_positive_real_guard_action_views"] for d in diagnoses),
            original_positive_real_guard_but_new_gain_nonpositive_action_views=sum(d["original_positive_real_guard_but_new_gain_nonpositive_action_views"] for d in diagnoses),
            switch_counts={k:sum(d["switch_counts"][k] for d in diagnoses) for k in api.POLICIES})))


def summarize(rows,cfg):
    sites=sorted({r["site"] for r in rows})
    expected={(f"single{f}_seed{seed}",site) for f in range(3) for seed in (17,29,43) for site in sites}
    if len(rows)!=54 or len(sites)!=6 or {(r["trial"],r["site"]) for r in rows}!=expected:raise ValueError("All54views required")
    ci=lambda fn:prior.metrics.interval(rows,fn,expected_sites=sites,resamples=cfg["bootstrap_resamples"],seed=cfg["bootstrap_seed"])
    accuracy={a:ci(lambda r,a=a:r["policies"][a]["known_ADE_gain_floor_percent"]) for a in api.POLICIES}
    effects={a:{k:ci(lambda r,a=a,k=k:r["effects"][a][k]) for k in ("known_pp","completion_lower_pp","completion_upper_pp")}
        for a in ("fixed_real_minus_fixed_noise","fixed_real_minus_coupled_disabled","fixed_real_minus_coupled_real")}
    safety={}
    for arm in api.POLICIES:
        views=[v for r in rows for v in r["policies"][arm]["recordings"].values()];easy=[v["known_easy_net_degradation"] for v in views if v["known_easy_net_degradation"] is not None]
        safety[arm]=dict(required_record_views=len(views),supported_record_views=sum(v["supported"] for v in views),
            known_all_risk_violations=sum(v["known_all_risk"] is not None and v["known_all_risk"]>.02 for v in views),
            known_easy_risk_violations=sum(v["known_easy_risk"] is not None and v["known_easy_risk"]>.02 for v in views),
            undefined_easy_views=sum(v["completion_easy_risk_upper"] is None for v in views),
            worst_known_easy_net_degradation_percent=100*max(easy) if easy else None)
    primary=effects["fixed_real_minus_fixed_noise"]["completion_lower_pp"]["CI95"][0]>0
    advancement=effects["fixed_real_minus_coupled_disabled"]["completion_lower_pp"]["CI95"][0]>0
    risk=safety["fixed_utility_real_guard"]["supported_record_views"]==252
    return dict(groups=54,unique_target_rows=38102,unique_recordings=28,selection_localities=6,
        accuracy=accuracy,effects=effects,safety=safety,primary_pass=primary,advancement_pass=advancement,
        original_record_risk_pass=risk,policy_candidate_pass=primary and advancement and risk,
        new_training=False,new_cost_inference=False,threshold_search=False,new_joint_solver=False,
        selection_development_not_confirmation=True,calibration_access=False,confirmation_access=False,
        deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)


def evaluate(verify=False,resume=False):
    cfg,rows,_=admission();beat,guard,receipt=runtime("decision_verify" if verify else "decision_eval")
    finish=PUBLIC/("verification.json" if verify else "complete.json")
    if finish.exists():raise ValueError("Complete decision readout immutable")
    frozen=json.loads((PUBLIC/"policy_freeze.json").read_text());saved={}
    if len(frozen["policies"])!=9:raise ValueError("Freeze all nine cases first")
    for ref in frozen["policies"]:
        v=json.loads(checked(ref).read_text());saved[v["trial"]]=np.load(checked(v["choices"]),allow_pickle=False)
    results=[];refs=[]
    for row in rows:
        guard();data,paths,choices,_,_=frozen_case(row)
        fresh=np.stack([choices[k] for k in api.POLICIES]);np.testing.assert_array_equal(saved[row["trial"]],fresh)
        target,valid=prior.selection_labels(data);floor=paths[:,row["baseline_index"]]
        policies={k:prior.selected_paths(paths,v,row["baseline_index"]) for k,v in choices.items()}
        for site in sorted(set(data["sites"])):
            at=data["sites"]==site
            scored={k:prior.metrics.policy(v[at],floor[at],paths[at,1],target[at],valid[at],data["recordings"][at],
                easy_cut=row["easy_cut"],hard_cut=row["hard_cut"]) for k,v in policies.items()}
            old=json.loads((parent.PUBLIC/"groups"/(row["trial"]+"_"+str(site)+".json")).read_text())
            for arm in ("disabled","real","noise"):
                if scored["coupled_"+arm]!=old["policies"][arm]:raise ValueError("Coupled control differs")
            comparisons={"fixed_real_minus_fixed_noise":("fixed_utility_noise_guard","fixed_utility_real_guard"),
                "fixed_real_minus_coupled_disabled":("coupled_disabled","fixed_utility_real_guard"),
                "fixed_real_minus_coupled_real":("coupled_real","fixed_utility_real_guard")}
            effects={name:prior.metrics.paired(policies[a][at],policies[b][at],floor[at],target[at],valid[at]) for name,(a,b) in comparisons.items()}
            value=dict(trial=row["trial"],seed=row["seed"],fold=row["fold"],site=str(site),policies=scored,effects=effects,
                policy_freeze_sha256=sha(PUBLIC/"policy_freeze.json"),registration_sha256=sha(REG))
            p=PUBLIC/"groups"/(row["trial"]+"_"+str(site)+".json")
            if verify or resume and p.exists():
                if json.loads(p.read_text())!=value:raise ValueError("Decision metric replay differs")
            else:once(p,value)
            results.append(value);refs.append(artifact(p))
        beat(state="decision_metric_group_complete",trial=row["trial"],groups=len(results),total=54)
    s=summarize(results,cfg)
    if verify:
        if json.loads((PUBLIC/"summary.json").read_text())!=s:raise ValueError("Decision summary replay differs")
        once(finish,dict(**receipt(),all9_choice_matrices_exact=True,all54_metric_groups_exact=True,new_training=False,
            new_cost_inference=False,summary_sha256=sha(PUBLIC/"summary.json"),complete_sha256=sha(PUBLIC/"complete.json")))
    else:
        once(PUBLIC/"summary.json",s);once(finish,dict(**receipt(),groups=refs,summary_sha256=sha(PUBLIC/"summary.json")))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("register","freeze","eval","verify"));parser.add_argument("--resume",action="store_true");args=parser.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=="register":admission(True)
        elif args.phase=="freeze":freeze(args.resume)
        else:evaluate(args.phase=="verify",args.resume)
