"""SOURCE-only factorial loss repair; exposed selection is not confirmation."""
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

if platform.system() == "Darwin" and platform.machine() != "arm64":
    raise RuntimeError("Native arm64 before Torch")
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key, "4")
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_selection_controller as prior
from scripts import run_m3w_selection_cost_transport as transport
from src.world_model import m3w_selection_cardinal_cost as api

NAME = "european_selection_cardinal_cost_v1"
PUBLIC = ROOT/"outputs/publication_readiness_2026_09"/NAME
PRIVATE = ROOT/"data/stage_cvpr2027_experiments"/NAME
CONFIG = ROOT/"configs/m3w_selection_cardinal_cost_v1.json"; REG = PUBLIC/"registration.json"
sha, artifact, checked, write, once = prior.sha, prior.artifact, prior.checked, prior.write, prior.once
FILES = (Path(__file__), CONFIG, ROOT/"src/world_model/m3w_selection_cardinal_cost.py",
         ROOT/"tests/test_m3w_selection_cardinal_cost.py", PUBLIC/"protocol.md")


def admission(register=False):
    cfg = json.loads(CONFIG.read_text()); parent, producers, ids = prior.admission()
    verified = json.loads((transport.PUBLIC/"verification.json").read_text())
    if (not verified["all18_SOURCE_cost_inferences_exact"] or not verified["all252_diagnostic_groups_exact"]
            or verified["summary_sha256"] != sha(transport.PUBLIC/"summary.json")):
        raise ValueError("Completed SOURCE/selection transport diagnosis required")
    if (cfg["new_arms"] != list(api.NEW_ARMS) or cfg["new_fits"] != 27
            or cfg["training"] != parent["training"] or cfg["positive_harm_budget"] != .02
            or cfg["easy_net_degradation_budget"] != .02
            or any(cfg[k] for k in ("model_selection_labels_for_training", "threshold_search", "calibration_access",
                "confirmation_access", "new_trajectory_training", "stage5c_executed", "smc_enabled"))):
        raise ValueError("Unchanged data/model/budget, fixed factorial loss arms required")
    dependencies = ("src/world_model/m3w_selection_controller.py", "src/world_model/m3w_multiaction_cost.py",
        "src/evaluation/m3w_selection_forecast_readout.py", "scripts/run_m3w_selection_controller.py")
    value = dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        dependencies={p:sha(ROOT/p) for p in dependencies},
        parent_registration=artifact(prior.REG), parent_training=artifact(prior.PUBLIC/"training_freeze.json"),
        parent_policy=artifact(prior.PUBLIC/"policy_freeze.json"), parent_verification=artifact(prior.PUBLIC/"verification.json"),
        diagnosis=artifact(transport.PUBLIC/"summary.json"), diagnosis_verification=artifact(transport.PUBLIC/"verification.json"),
        historical_failed_easy_deviance=artifact(ROOT/"outputs/publication_readiness_2026_09/european_easy_harm_deviance_v1/readout_verified_v3/conclusions.md"),
        new_fits=27, cached_control_fits=9, source_whitelist_rows=len(ids), primary=cfg["primary"],
        actual_seeds=[17,29,43], selection_already_exposed=True, new_joint_solver=False,
        model_selection_labels_for_training=False, threshold_search=False, calibration_access=False,
        confirmation_access=False, stage5c_executed=False, smc_enabled=False)
    if register: once(REG, value)
    if json.loads(REG.read_text()) != value: raise ValueError("Registered loss comparison changed")
    if not register:
        for p in (*FILES, REG):
            blob = subprocess.run(["git","show","HEAD:"+str(p.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
            if hashlib.sha256(blob).hexdigest() != sha(p): raise ValueError("Commit contract before real execution")
    return cfg, producers, ids


def runtime(cfg, phase):
    torch.set_num_threads(4); torch.set_num_interop_threads(1); start = time.monotonic()
    def interrupt(*_): raise KeyboardInterrupt("Atomic checkpoints allow unchanged resume")
    signal.signal(signal.SIGTERM, interrupt)
    def guard():
        if shutil.disk_usage(ROOT).free < cfg["disk_reserve_bytes"]: raise OSError("Preserve10GiB free")
        if time.monotonic()-start > cfg["hard_runtime_limit_seconds"]: raise TimeoutError("12hboundary;resume")
    def beat(**kw):
        guard(); row = dict(pid=os.getpid(),utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),phase=phase,**kw)
        write(PRIVATE/"heartbeat.json",row);print(json.dumps(row),flush=True)
    def receipt(): return dict(pid=os.getpid(),seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,registration_sha256=sha(REG))
    beat(state="runtime_ready",architecture=platform.machine(),torch=torch.__version__,device="cpu",threads=4,interop=1,workers=0)
    return beat,guard,receipt


def fit_path(row, arm, family="fits"): return PRIVATE/family/(row["trial"]+"_"+arm+".pt.gz")


def one(data, row, arm, cfg, beat, guard, *, family="fits", resume=False, stop_at=None):
    return api.fit(data,arm=arm,seed=row["seed"],settings=cfg["training"],
        identity=dict(registration_sha256=sha(REG),source_bank=artifact(prior.PRIVATE/"source_banks"/row["trial"]/"complete.json"),
                      producer=row["checkpoint"]),path=fit_path(row,arm,family),
        heartbeat=lambda **v:beat(trial=row["trial"],**v),guard=guard,resume=resume,stop_at=stop_at)


MATCHED = ("initial_model","parameters","budget","sampler","draw_hash","fit_ids","preprocess","torch_rng")


def pilot():
    cfg,producers,ids = admission();beat,guard,receipt = runtime(cfg,"pilot")
    if (PUBLIC/"pilot.json").exists(): raise ValueError("Completed pilot immutable")
    source = prior.source_inputs(json.loads(prior.frozen.REG.read_text()));row = producers[0]
    data = prior.fitting_data(row,source,ids);cells=[]
    for arm in api.ARMS:
        one(data,row,arm,cfg,beat,guard,family="pilot_resume",stop_at=50)
        a = one(data,row,arm,cfg,beat,guard,family="pilot_resume",resume=True,stop_at=100)
        b = one(data,row,arm,cfg,beat,guard,family="pilot_direct",stop_at=100)
        prior.assert_recovery(a,b)
        original_exact = None
        if arm == "weighted_centered":
            original = prior.api.read_checkpoint(prior.fit_path(row["trial"],"context","pilot_direct_deterministic_v2"))
            for key in (*MATCHED,"model","optimizer","trace","step"): prior.api.exact(b[key],original[key])
            original_exact = True
        cells.append(dict(arm=arm,updates=100,parameters=b["parameters"],seconds=b["seconds"],budget=b["budget"],
            exact_recovery=True,original_control_100_step_exact=original_exact))
    once(PUBLIC/"pilot.json",dict(**receipt(),engineering_pass=True,cells=cells,optimizer_updates=800,
        source_rows=len(data["causal"]["ids"]),selection_labels_read=False,
        projected27_primary_fit_seconds=sum(c["seconds"]/100*c["budget"] for c in cells[1:])*9,
        projection_excludes_data_preparation_and_full_refits=True))


def train(resume):
    cfg,producers,ids = admission();beat,guard,receipt = runtime(cfg,"train")
    if (PUBLIC/"training_freeze.json").exists(): raise ValueError("Training endpoints immutable")
    if not json.loads((PUBLIC/"pilot.json").read_text())["engineering_pass"]: raise ValueError("Real recovery pilot required")
    source=prior.source_inputs(json.loads(prior.frozen.REG.read_text()));parents=prior.fitted();refs=[];updates=refits=0
    for row in producers:
        data=prior.fitting_data(row,source,ids);control=prior.api.read_checkpoint(checked(parents[row["trial"],"context"]["checkpoint"]))
        for arm in api.NEW_ARMS:
            state=one(data,row,arm,cfg,beat,guard,resume=resume and fit_path(row,arm).exists())
            for key in MATCHED: prior.api.exact(state[key],control[key])
            if state["step"]!=state["budget"] or state["sampler"]["counts"].min()<4:
                raise ValueError("Complete matched query coverage required")
            if row==producers[0]:
                other=one(data,row,arm,cfg,beat,guard,family="full_refit",resume=resume and fit_path(row,arm,"full_refit").exists())
                prior.assert_recovery(state,other);refits+=state["step"]
                once(PUBLIC/("first_full_refit_"+arm+".json"),dict(scientific_state_exact=True,updates=state["step"],
                    checkpoint=artifact(fit_path(row,arm)),refit=artifact(fit_path(row,arm,"full_refit"))))
            path=PUBLIC/"fits"/(row["trial"]+"_"+arm+".json")
            value=dict(trial=row["trial"],seed=row["seed"],fold=row["fold"],arm=arm,checkpoint=artifact(fit_path(row,arm)),
                step=state["step"],budget=state["budget"],parameters=state["parameters"],seconds=state["seconds"],
                source_sites=state["source_sites"],producer_sites=row["fit_sites"],fitting_rows=len(state["fit_ids"]),
                fit_ids_sha256=prior.frozen.fingerprint(state["fit_ids"]),draw_hash=state["draw_hash"],
                minimum_query_visits=int(state["sampler"]["counts"].min()),unknown_supervised_rows=0,
                first_trace=state["trace"][0],final_trace=state["trace"][-1],matched_to_original_context_exact=True,
                registration_sha256=sha(REG))
            once(path,value);refs.append(artifact(path));updates+=state["step"]
            beat(state="fit_complete",trial=row["trial"],arm=arm,fits=len(refs),total=27)
        del data,control,state
    once(PUBLIC/"training_freeze.json",dict(**receipt(),fits=refs,primary_optimizer_updates=updates,
        exact_full_refit_updates=refits,source_only=True,selection_labels_used=False))


def fitted():
    values={}
    for ref in json.loads((PUBLIC/"training_freeze.json").read_text())["fits"]:
        row=json.loads(checked(ref).read_text());checked(row["checkpoint"])
        if row["step"]!=row["budget"] or row["minimum_query_visits"]<4:raise ValueError("Complete endpoints required")
        values[row["trial"],row["arm"]]=row
    if len(values)!=27:raise ValueError("All27fits required")
    return values


def infer(resume):
    cfg,producers,_=admission();beat,guard,receipt=runtime(cfg,"infer");models=fitted();refs=[]
    if (PUBLIC/"policy_freeze.json").exists():raise ValueError("Frozen actions immutable")
    for row in producers:
        data,_,causal=prior.selection_bank(row)
        for arm in api.NEW_ARMS:
            guard();path=PUBLIC/"policies"/(row["trial"]+"_"+arm+".json")
            if resume and path.exists():
                ref=json.loads(path.read_text());checked(ref["prediction"]);checked(ref["choices"])
                if ref["registration_sha256"]!=sha(REG):raise ValueError("Policy binding changed")
                refs.append(artifact(path));continue
            state=prior.api.read_checkpoint(checked(models[row["trial"],arm]["checkpoint"]))
            pred=prior.api.predict(state,causal);choices,_=prior.allocator.select_actions(pred,causal["allowed"],arm="physical_neural")
            home=PRIVATE/"policies"/(row["trial"]+"_"+arm);home.mkdir(parents=True,exist_ok=True)
            prior.frozen.inputs.store_array(home/"prediction.npy",pred);prior.frozen.inputs.store_array(home/"choices.npy",choices)
            once(path,dict(trial=row["trial"],arm=arm,prediction=artifact(home/"prediction.npy"),choices=artifact(home/"choices.npy"),
                checkpoint=models[row["trial"],arm]["checkpoint"],inference_features=prior.frozen.fingerprint(causal["x"]),
                registration_sha256=sha(REG),future_labels_read=False,rows=len(data["ids"])))
            refs.append(artifact(path));beat(state="policy_frozen",trial=row["trial"],arm=arm,policies=len(refs),total=27)
    once(PUBLIC/"policy_freeze.json",dict(**receipt(),policies=refs,training_freeze=artifact(PUBLIC/"training_freeze.json"),
        labels_read_for_this_comparison=False,threshold_search=False))


def summarize(rows,cfg):
    sites=sorted({r["site"] for r in rows})
    expected={(f"single{f}_seed{s}",site) for f in range(3) for s in (17,29,43) for site in sites}
    if len(rows)!=54 or len(sites)!=6 or {(r["trial"],r["site"]) for r in rows}!=expected:
        raise ValueError("All9producers by six sites required")
    ci=lambda fn:prior.metrics.interval(rows,fn,expected_sites=sites,resamples=cfg["bootstrap_resamples"],seed=cfg["bootstrap_seed"])
    accuracy={a:ci(lambda r,a=a:r["policies"][a]["known_ADE_gain_floor_percent"]) for a in api.ARMS}
    effects={a:{k:ci(lambda r,a=a,k=k:r["effects"][a][k]) for k in ("known_pp","completion_lower_pp","completion_upper_pp")}
             for a in api.NEW_ARMS}
    safety={}
    for arm in api.ARMS:
        views=[v for r in rows for v in r["policies"][arm]["recordings"].values()]
        easy=[v["known_easy_net_degradation"] for v in views if v["known_easy_net_degradation"] is not None]
        safety[arm]=dict(required_record_views=len(views),supported_record_views=sum(v["supported"] for v in views),
            known_all_risk_violations=sum(v["known_all_risk"] is not None and v["known_all_risk"]>.02 for v in views),
            known_easy_risk_violations=sum(v["known_easy_risk"] is not None and v["known_easy_risk"]>.02 for v in views),
            undefined_easy_views=sum(v["completion_easy_risk_upper"] is None for v in views),
            worst_known_easy_net_degradation_percent=100*max(easy) if easy else None)
    primary_pass=effects["mean_absolute"]["completion_lower_pp"]["CI95"][0]>0
    risk_pass=safety["mean_absolute"]["supported_record_views"]==safety["mean_absolute"]["required_record_views"]
    return dict(groups=54,unique_target_rows=38102,unique_recordings=28,selection_localities=6,
        accuracy=accuracy,effects=effects,safety=safety,primary_pass=primary_pass,original_record_risk_pass=risk_pass,
        policy_candidate_pass=primary_pass and risk_pass,selection_development_not_confirmation=True,
        selection_labels_used_for_training=False,threshold_search=False,calibration_access=False,confirmation_access=False,
        new_joint_solver=False,deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)


def evaluate(verify=False,resume=False):
    cfg,producers,_=admission();beat,guard,receipt=runtime(cfg,"verify" if verify else "eval")
    finish=PUBLIC/("verification.json" if verify else "complete.json")
    if finish.exists():raise ValueError("Complete readout immutable")
    freeze=json.loads((PUBLIC/"policy_freeze.json").read_text());saved={}
    if freeze["training_freeze"]!=artifact(PUBLIC/"training_freeze.json") or len(freeze["policies"])!=27:
        raise ValueError("All27policies must freeze before outcomes")
    for ref in freeze["policies"]:
        value=json.loads(checked(ref).read_text());checked(value["prediction"]);checked(value["choices"])
        saved[value["trial"],value["arm"]]=value
    refs=[];rows=[]
    for row in producers:
        guard();data,paths,causal=prior.selection_bank(row);target,valid=prior.selection_labels(data);floor=paths[:,row["baseline_index"]]
        control=json.loads((prior.PUBLIC/"policies"/(row["trial"]+"_context.json")).read_text())
        cp=np.load(checked(control["choices"]),allow_pickle=False)[control["policy_names"].index("independent")]
        policies={"weighted_centered":prior.selected_paths(paths,cp,row["baseline_index"])}
        for arm in api.NEW_ARMS:
            ref=saved[row["trial"],arm];pred=np.load(checked(ref["prediction"]),allow_pickle=False)
            choices=np.load(checked(ref["choices"]),allow_pickle=False)
            if verify:
                fresh=prior.api.predict(prior.api.read_checkpoint(checked(ref["checkpoint"])),causal)
                np.testing.assert_array_equal(fresh,pred)
            fresh_choices,_=prior.allocator.select_actions(pred,causal["allowed"],arm="physical_neural")
            np.testing.assert_array_equal(fresh_choices,choices)
            policies[arm]=prior.selected_paths(paths,choices,row["baseline_index"])
        for site in sorted(set(data["sites"])):
            at=data["sites"]==site
            scored={k:prior.metrics.policy(v[at],floor[at],paths[at,1],target[at],valid[at],data["recordings"][at],
                easy_cut=row["easy_cut"],hard_cut=row["hard_cut"]) for k,v in policies.items()}
            old=json.loads((prior.PUBLIC/"groups"/(row["trial"]+"_"+str(site)+".json")).read_text())
            if scored["weighted_centered"]!=old["policies"]["context_independent"]:raise ValueError("Cached control changed")
            effects={a:prior.metrics.paired(policies["weighted_centered"][at],policies[a][at],floor[at],target[at],valid[at]) for a in api.NEW_ARMS}
            value=dict(trial=row["trial"],seed=row["seed"],fold=row["fold"],site=str(site),policies=scored,effects=effects,
                registration_sha256=sha(REG),policy_freeze_sha256=sha(PUBLIC/"policy_freeze.json"))
            path=PUBLIC/"groups"/(row["trial"]+"_"+str(site)+".json")
            if verify or resume and path.exists():
                if json.loads(path.read_text())!=value:raise ValueError("Scalar replay differs")
            else:once(path,value)
            refs.append(artifact(path));rows.append(value)
        beat(state="readout_group_complete",trial=row["trial"],groups=len(rows),total=54)
    summary=summarize(rows,cfg)
    if verify:
        if summary!=json.loads((PUBLIC/"summary.json").read_text()):raise ValueError("Full summary replay differs")
        once(finish,dict(**receipt(),all27_new_inferences_exact=True,all54_metric_groups_exact=True,
            cached_control_exact=True,summary_sha256=sha(PUBLIC/"summary.json"),complete_sha256=sha(PUBLIC/"complete.json")))
    else:
        once(PUBLIC/"summary.json",summary);once(finish,dict(**receipt(),groups=refs,policy_freeze=artifact(PUBLIC/"policy_freeze.json"),
            summary_sha256=sha(PUBLIC/"summary.json")))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("phase",choices=("register","pilot","train","infer","eval","verify"))
    parser.add_argument("--resume",action="store_true");args=parser.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=="register":admission(True)
        elif args.phase=="pilot":pilot()
        elif args.phase=="train":train(args.resume)
        elif args.phase=="infer":infer(args.resume)
        else:evaluate(args.phase=="verify",args.resume)
