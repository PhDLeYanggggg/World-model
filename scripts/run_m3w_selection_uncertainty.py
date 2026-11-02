"""SOURCE-only seed-spread input comparison on the unchanged cardinal critic."""
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
import torch
from scripts import run_m3w_selection_cardinal_cost as parent
from src.world_model import m3w_selection_uncertainty as api

prior=parent.prior
NAME="european_selection_uncertainty_v1"
PUBLIC=ROOT/"outputs/publication_readiness_2026_09"/NAME
PRIVATE=ROOT/"data/stage_cvpr2027_experiments"/NAME
CONFIG=ROOT/"configs/m3w_selection_uncertainty_v1.json";REG=PUBLIC/"registration.json"
sha,artifact,checked,write,once=prior.sha,prior.artifact,prior.checked,prior.write,prior.once
FILES=(Path(__file__),CONFIG,ROOT/"src/world_model/m3w_selection_uncertainty.py",
       ROOT/"tests/test_m3w_selection_uncertainty.py",PUBLIC/"protocol.md")


def admission(register=False):
    cfg=json.loads(CONFIG.read_text());old,producers,ids=parent.admission()
    v=json.loads((parent.PUBLIC/"verification.json").read_text())
    if not v["all27_new_inferences_exact"] or not v["all54_metric_groups_exact"] or v["summary_sha256"]!=sha(parent.PUBLIC/"summary.json"):
        raise ValueError("Complete previous cardinal-loss result required")
    if (cfg["arms"]!=list(api.ARMS) or cfg["fits"]!=27 or cfg["members_per_fold"]!=3
            or cfg["uncertainty_fields"]!=16 or cfg["training"]!=old["training"]
            or cfg["positive_harm_budget"]!=.02 or cfg["easy_net_degradation_budget"]!=.02
            or any(cfg[k] for k in ("new_forecast_training","threshold_search","selection_labels_for_training",
                "calibration_access","confirmation_access","stage5c_executed","smc_enabled"))):
        raise ValueError("Fixed SOURCE/selection task,loss,budget and risk required")
    scopes=json.loads(prior.REG.read_text())["fit_scopes"];folds=[]
    for fold in range(3):
        rows=[r for r in producers if r["fold"]==fold];ss=[r for r in scopes if r["trial"] in {v["trial"] for v in rows}]
        if (sorted(r["seed"] for r in rows)!=[17,29,43] or len({tuple(r["fit_sites"]) for r in rows})!=1
                or len({r["controller_fit_ids_sha256"] for r in ss})!=1):
            raise ValueError("Same-fold actual-seed ancestry and row alignment required")
        folds.append(dict(fold=fold,members=[r["trial"] for r in rows],
            producer_sites=rows[0]["fit_sites"],cost_sites=ss[0]["controller_fit_sites"],
            source_ids_sha256=ss[0]["controller_fit_ids_sha256"],rows=ss[0]["controller_fit_rows"]))
    dependencies=("src/world_model/m3w_selection_controller.py","src/world_model/m3w_selection_cardinal_cost.py",
        "src/evaluation/m3w_selection_forecast_readout.py","scripts/run_m3w_selection_controller.py",
        "scripts/run_m3w_selection_cardinal_cost.py")
    value=dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},dependencies={p:sha(ROOT/p) for p in dependencies},
        cardinal_registration=artifact(parent.REG),cardinal_training=artifact(parent.PUBLIC/"training_freeze.json"),
        cardinal_policies=artifact(parent.PUBLIC/"policy_freeze.json"),cardinal_verification=artifact(parent.PUBLIC/"verification.json"),
        cardinal_summary=artifact(parent.PUBLIC/"summary.json"),source_forecasts=artifact(prior.PUBLIC/"source_bank_freeze.json"),
        selection_forecasts=artifact(prior.frozen.PUBLIC/"prediction_freeze.json"),folds=folds,
        fields=list(api.FEATURES),fits=27,selection_already_exposed=True,new_joint_solver=False,
        selection_labels_for_training=False,threshold_search=False,calibration_access=False,confirmation_access=False,
        stage5c_executed=False,smc_enabled=False)
    if register:once(REG,value)
    if json.loads(REG.read_text())!=value:raise ValueError("Registered input test changed")
    if not register:
        for p in (*FILES,REG):
            blob=subprocess.run(["git","show","HEAD:"+str(p.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
            if hashlib.sha256(blob).hexdigest()!=sha(p):raise ValueError("Commit feature test before execution")
    return cfg,producers,ids


def runtime(cfg,phase):
    torch.set_num_threads(4);torch.set_num_interop_threads(1);start=time.monotonic()
    def interrupt(*_):raise KeyboardInterrupt("Resume last completed atomic checkpoint")
    signal.signal(signal.SIGTERM,interrupt)
    def guard():
        if shutil.disk_usage(ROOT).free<cfg["disk_reserve_bytes"]:raise OSError("Preserve10GiB reserve")
        if time.monotonic()-start>cfg["hard_runtime_limit_seconds"]:raise TimeoutError("12hboundary;resume")
    def beat(**kw):
        guard();r=dict(pid=os.getpid(),utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),phase=phase,**kw)
        write(PRIVATE/"heartbeat.json",r);print(json.dumps(r),flush=True)
    def receipt():return dict(pid=os.getpid(),seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,registration_sha256=sha(REG))
    beat(state="runtime_ready",architecture=platform.machine(),torch=torch.__version__,device="cpu",threads=4,interop=1,workers=0)
    return beat,guard,receipt


def member_prediction(row,domain):
    if domain=="source":
        meta=json.loads((prior.PRIVATE/"source_banks"/row["trial"]/"complete.json").read_text())
        if meta["binding"]["checkpoint"]!=row["checkpoint"]:raise ValueError("SOURCE forecast ancestry changed")
        return meta["prediction"],meta["ids_sha256"]
    frozen=json.loads((prior.frozen.PUBLIC/"prediction_freeze.json").read_text())
    meta=next(json.loads(checked(r).read_text()) for r in frozen["models"] if Path(r["path"]).stem==row["trial"]+"_history")
    return meta["prediction"],None


def build_features(resume=False,verify=False):
    cfg,producers,whitelist=admission();beat,guard,receipt=runtime(cfg,"feature_verify" if verify else "features")
    finish=PUBLIC/("feature_verification.json" if verify else "feature_freeze.json")
    if finish.exists():raise ValueError("Completed feature stage immutable")
    source=prior.source_inputs(json.loads(prior.frozen.REG.read_text()));selection=prior.frozen.load_features()
    predmeta=json.loads((prior.frozen.PUBLIC/"prediction_freeze.json").read_text())
    physical=np.load(checked(predmeta["physical_baselines"]),mmap_mode="r",allow_pickle=False);refs=[]
    for fold in range(3):
        rows=[r for r in producers if r["fold"]==fold];row=rows[0]
        for domain in ("source","selection"):
            guard();path=PUBLIC/"features"/(f"fold{fold}_{domain}.json")
            if resume and path.exists() and not verify:
                ref=json.loads(path.read_text());checked(ref["array"])
                if ref["registration_sha256"]!=sha(REG):raise ValueError("Feature checkpoint changed")
                refs.append(artifact(path));continue
            if domain=="source":
                ids=np.asarray(whitelist[~np.isin(source["sites"][whitelist],row["fit_sites"])])
                g=source["geometry"][ids];origin=source["origin"][ids]
                floor=prior.baseline_numpy(source["history"][ids],row["baseline_index"])-origin[:,None]
                sites=source["sites"][ids]
                if set(sites)&set(row["fit_sites"]):raise ValueError("Every ensemble producer must exclude cost rows")
            else:
                ids=selection["ids"];g=selection["geometry"];origin=selection["origin"]
                floor=physical[:,row["baseline_index"]]-origin[:,None];sites=selection["sites"]
            _,scales=prior.api.causal_cost_features(g,floor,floor);scales=scales.astype(np.float32)
            members=[];assets=[]
            for member in rows:
                ref,id_hash=member_prediction(member,domain)
                if domain=="source" and id_hash!=prior.frozen.fingerprint(ids):raise ValueError("Ensemble SOURCE rows differ")
                p=np.load(checked(ref),mmap_mode="r",allow_pickle=False)
                if p.shape!=(len(ids),12,2):raise ValueError("Aligned three-member forecast required")
                members.append(p if domain=="source" else p-origin[:,None]);assets.append(ref)
            u=api.spread_features(np.stack(members,1),scales)
            target=PRIVATE/"features"/(f"fold{fold}_{domain}.npy");target.parent.mkdir(exist_ok=True)
            if verify:np.testing.assert_array_equal(np.load(target,allow_pickle=False),u)
            else:prior.frozen.inputs.store_array(target,u)
            value=dict(fold=fold,domain=domain,rows=len(ids),fields=16,array=artifact(target),
                ids_sha256=prior.frozen.fingerprint(ids),scale_sha256=prior.frozen.fingerprint(scales),
                sites=sorted(set(sites)),producer_sites=row["fit_sites"],member_predictions=assets,
                finite=True,future_outcomes_used=False,normalization_fitted=False,registration_sha256=sha(REG))
            if verify:
                if json.loads(path.read_text())!=value:raise ValueError("Feature reconstruction differs")
            else:once(path,value)
            refs.append(artifact(path));beat(state="feature_frozen",fold=fold,domain=domain,rows=len(ids),groups=len(refs),total=6)
    if verify:
        if refs!=json.loads((PUBLIC/"feature_freeze.json").read_text())["features"]:raise ValueError("Feature roster differs")
        once(finish,dict(**receipt(),all6_feature_arrays_exact=True,feature_freeze=artifact(PUBLIC/"feature_freeze.json")))
    else:once(finish,dict(**receipt(),features=refs,future_outcomes_used=False,new_forecasts_trained=False))


def load_u(fold,domain,causal):
    path=PUBLIC/"features"/(f"fold{fold}_{domain}.json");ref=json.loads(path.read_text())
    if (ref["registration_sha256"]!=sha(REG) or ref["ids_sha256"]!=prior.frozen.fingerprint(causal["ids"])
            or ref["scale_sha256"]!=prior.frozen.fingerprint(causal["scales"])):raise ValueError("Feature row/scale alignment differs")
    return np.load(checked(ref["array"]),mmap_mode="r",allow_pickle=False)


def fit_path(row,arm,family="fits"):return PRIVATE/family/(row["trial"]+"_"+arm+".pt.gz")


def one(data,row,arm,cfg,beat,guard,*,family="fits",resume=False,stop_at=None):
    return api.fit(data,arm=arm,seed=row["seed"],settings=cfg["training"],
        identity=dict(registration_sha256=sha(REG),source_features=artifact(PUBLIC/"features"/(f"fold{row['fold']}_source.json")),
            producer=row["checkpoint"]),path=fit_path(row,arm,family),
        heartbeat=lambda **v:beat(trial=row["trial"],**v),guard=guard,resume=resume,stop_at=stop_at)


def fitting_data(row,source,ids):
    d=prior.fitting_data(row,source,ids)
    return api.augment_data(d,load_u(row["fold"],"source",d["causal"]))


def pilot():
    cfg,producers,ids=admission();beat,guard,receipt=runtime(cfg,"pilot")
    if (PUBLIC/"pilot.json").exists():raise ValueError("Complete pilot immutable")
    if not json.loads((PUBLIC/"feature_verification.json").read_text())["all6_feature_arrays_exact"]:raise ValueError("Feature replay first")
    source=prior.source_inputs(json.loads(prior.frozen.REG.read_text()));row=producers[0];data=fitting_data(row,source,ids);cells=[]
    for arm in api.ARMS:
        one(data,row,arm,cfg,beat,guard,family="pilot_resume",stop_at=50)
        a=one(data,row,arm,cfg,beat,guard,family="pilot_resume",resume=True,stop_at=100)
        b=one(data,row,arm,cfg,beat,guard,family="pilot_direct",stop_at=100);prior.assert_recovery(a,b)
        if arm=="disabled":
            old=prior.api.read_checkpoint(parent.fit_path(row,"mean_absolute","pilot_direct"));api.compare_disabled(b,old)
        cells.append(dict(arm=arm,exact_recovery=True,updates=100,parameters=b["parameters"],
            seconds=b["seconds"],budget=b["budget"],original_control_exact=arm=="disabled"))
    once(PUBLIC/"pilot.json",dict(**receipt(),engineering_pass=True,cells=cells,optimizer_updates=600,
        projected27_primary_fit_seconds=sum(r["seconds"]*r["budget"]/100 for r in cells)*9,
        projection_excludes_preparation_and_full_refits=True,selection_outcomes_scored=False))


MATCHED=("initial_model","parameters","budget","sampler","draw_hash","fit_ids","preprocess","uncertainty_preprocess","torch_rng")


def train(resume=False):
    cfg,producers,ids=admission();beat,guard,receipt=runtime(cfg,"train")
    if (PUBLIC/"training_freeze.json").exists():raise ValueError("Complete fits immutable")
    if not json.loads((PUBLIC/"pilot.json").read_text())["engineering_pass"]:raise ValueError("Real pilot required")
    source=prior.source_inputs(json.loads(prior.frozen.REG.read_text()));oldfits=parent.fitted();refs=[];updates=refits=0
    for row in producers:
        data=fitting_data(row,source,ids);states=[]
        for arm in api.ARMS:
            state=one(data,row,arm,cfg,beat,guard,resume=resume and fit_path(row,arm).exists())
            if state["step"]!=state["budget"] or state["sampler"]["counts"].min()<4:raise ValueError("Complete coverage required")
            if states:
                for key in MATCHED:api.base.exact(state[key],states[0][key])
            if arm=="disabled":api.compare_disabled(state,prior.api.read_checkpoint(checked(oldfits[row["trial"],"mean_absolute"]["checkpoint"])))
            if row==producers[0]:
                other=one(data,row,arm,cfg,beat,guard,family="full_refit",resume=resume and fit_path(row,arm,"full_refit").exists())
                prior.assert_recovery(state,other);refits+=state["step"]
                once(PUBLIC/("first_full_refit_"+arm+".json"),dict(scientific_state_exact=True,updates=state["step"],
                    checkpoint=artifact(fit_path(row,arm)),refit=artifact(fit_path(row,arm,"full_refit"))))
            p=PUBLIC/"fits"/(row["trial"]+"_"+arm+".json")
            once(p,dict(trial=row["trial"],seed=row["seed"],fold=row["fold"],arm=arm,checkpoint=artifact(fit_path(row,arm)),
                parameters=state["parameters"],step=state["step"],budget=state["budget"],seconds=state["seconds"],
                source_sites=state["source_sites"],producer_sites=row["fit_sites"],fitting_rows=len(state["fit_ids"]),
                fit_ids_sha256=prior.frozen.fingerprint(state["fit_ids"]),draw_hash=state["draw_hash"],
                minimum_query_visits=int(state["sampler"]["counts"].min()),unknown_supervised_rows=0,
                catalogue_sha256=prior.frozen.fingerprint(state["uncertainty_preprocess"]["catalog"]),
                first_trace=state["trace"][0],final_trace=state["trace"][-1],registration_sha256=sha(REG)))
            refs.append(artifact(p));updates+=state["step"];states.append(state)
            beat(state="fit_complete",trial=row["trial"],arm=arm,fits=len(refs),total=27)
        del data,states
    once(PUBLIC/"training_freeze.json",dict(**receipt(),fits=refs,primary_optimizer_updates=updates,
        exact_full_refit_updates=refits,all9_disabled_original_controls_exact=True,source_only=True,selection_labels_used=False))


def fitted():
    out={}
    for ref in json.loads((PUBLIC/"training_freeze.json").read_text())["fits"]:
        row=json.loads(checked(ref).read_text());checked(row["checkpoint"])
        if row["step"]!=row["budget"] or row["minimum_query_visits"]<4:raise ValueError("Complete source fit required")
        out[row["trial"],row["arm"]]=row
    if len(out)!=27:raise ValueError("All27fit endpoints required")
    return out


def infer(resume=False):
    cfg,producers,_=admission();beat,guard,receipt=runtime(cfg,"infer");models=fitted();refs=[]
    if (PUBLIC/"policy_freeze.json").exists():raise ValueError("Complete policy freeze immutable")
    for row in producers:
        data,_,causal=prior.selection_bank(row);u=load_u(row["fold"],"selection",causal)
        for arm in api.ARMS:
            guard();p=PUBLIC/"policies"/(row["trial"]+"_"+arm+".json")
            if resume and p.exists():
                value=json.loads(p.read_text());checked(value["prediction"]);checked(value["choices"])
                if value["registration_sha256"]!=sha(REG):raise ValueError("Frozen policy identity changed")
                refs.append(artifact(p));continue
            cp=prior.api.read_checkpoint(checked(models[row["trial"],arm]["checkpoint"]))
            pred=api.predict(cp,causal,u,domain="selection");choices,_=prior.allocator.select_actions(pred,causal["allowed"],arm="physical_neural")
            if arm=="disabled":
                old=json.loads((parent.PUBLIC/"policies"/(row["trial"]+"_mean_absolute.json")).read_text())
                np.testing.assert_array_equal(pred,np.load(checked(old["prediction"]),allow_pickle=False))
                np.testing.assert_array_equal(choices,np.load(checked(old["choices"]),allow_pickle=False))
            home=PRIVATE/"policies"/(row["trial"]+"_"+arm);home.mkdir(parents=True,exist_ok=True)
            prior.frozen.inputs.store_array(home/"prediction.npy",pred);prior.frozen.inputs.store_array(home/"choices.npy",choices)
            once(p,dict(trial=row["trial"],arm=arm,prediction=artifact(home/"prediction.npy"),choices=artifact(home/"choices.npy"),
                checkpoint=models[row["trial"],arm]["checkpoint"],registration_sha256=sha(REG),future_labels_read=False,
                inference_features=prior.frozen.fingerprint(causal["x"]),uncertainty_features=prior.frozen.fingerprint(u),rows=len(data["ids"])))
            refs.append(artifact(p));beat(state="policy_frozen",trial=row["trial"],arm=arm,policies=len(refs),total=27)
    once(PUBLIC/"policy_freeze.json",dict(**receipt(),policies=refs,training_freeze=artifact(PUBLIC/"training_freeze.json"),
        all9_disabled_policy_predictions_exact=True,labels_read_for_this_comparison=False,threshold_search=False))


def summarize(rows,cfg):
    sites=sorted({r["site"] for r in rows});expected={(f"single{f}_seed{s}",site) for f in range(3) for s in (17,29,43) for site in sites}
    if len(rows)!=54 or len(sites)!=6 or {(r["trial"],r["site"]) for r in rows}!=expected:raise ValueError("All54views required")
    ci=lambda fn:prior.metrics.interval(rows,fn,expected_sites=sites,resamples=cfg["bootstrap_resamples"],seed=cfg["bootstrap_seed"])
    accuracy={a:ci(lambda r,a=a:r["policies"][a]["known_ADE_gain_floor_percent"]) for a in api.ARMS}
    effects={a:{k:ci(lambda r,a=a,k=k:r["effects"][a][k]) for k in ("known_pp","completion_lower_pp","completion_upper_pp")}
        for a in ("real_minus_disabled","real_minus_noise")}
    safety={}
    for arm in api.ARMS:
        views=[v for r in rows for v in r["policies"][arm]["recordings"].values()]
        easy=[v["known_easy_net_degradation"] for v in views if v["known_easy_net_degradation"] is not None]
        safety[arm]=dict(required_record_views=len(views),supported_record_views=sum(v["supported"] for v in views),
            known_all_risk_violations=sum(v["known_all_risk"] is not None and v["known_all_risk"]>.02 for v in views),
            known_easy_risk_violations=sum(v["known_easy_risk"] is not None and v["known_easy_risk"]>.02 for v in views),
            undefined_easy_views=sum(v["completion_easy_risk_upper"] is None for v in views),
            worst_known_easy_net_degradation_percent=100*max(easy) if easy else None)
    primary=effects["real_minus_disabled"]["completion_lower_pp"]["CI95"][0]>0
    noise=effects["real_minus_noise"]["completion_lower_pp"]["CI95"][0]>0
    risk=safety["real"]["supported_record_views"]==safety["real"]["required_record_views"]
    return dict(groups=54,unique_target_rows=38102,unique_recordings=28,selection_localities=6,
        accuracy=accuracy,effects=effects,safety=safety,primary_pass=primary,negative_control_pass=noise,
        original_record_risk_pass=risk,policy_candidate_pass=primary and noise and risk,
        selection_development_not_final_confirmation=True,selection_labels_used_for_training=False,
        threshold_search=False,new_joint_solver=False,calibration_access=False,confirmation_access=False,
        deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)


def evaluate(verify=False,resume=False):
    cfg,producers,_=admission();beat,guard,receipt=runtime(cfg,"verify" if verify else "eval")
    finish=PUBLIC/("verification.json" if verify else "complete.json")
    if finish.exists():raise ValueError("Complete readout immutable")
    freeze=json.loads((PUBLIC/"policy_freeze.json").read_text());saved={}
    if len(freeze["policies"])!=27 or freeze["training_freeze"]!=artifact(PUBLIC/"training_freeze.json"):
        raise ValueError("All27policies must freeze before scoring")
    for ref in freeze["policies"]:
        value=json.loads(checked(ref).read_text());checked(value["prediction"]);checked(value["choices"])
        saved[value["trial"],value["arm"]]=value
    refs=[];rows=[]
    for row in producers:
        guard();data,paths,causal=prior.selection_bank(row);u=load_u(row["fold"],"selection",causal)
        target,valid=prior.selection_labels(data);floor=paths[:,row["baseline_index"]];policies={}
        for arm in api.ARMS:
            ref=saved[row["trial"],arm];pred=np.load(checked(ref["prediction"]),allow_pickle=False);choices=np.load(checked(ref["choices"]),allow_pickle=False)
            if verify:
                fresh=api.predict(prior.api.read_checkpoint(checked(ref["checkpoint"])),causal,u,domain="selection")
                np.testing.assert_array_equal(fresh,pred)
            actual,_=prior.allocator.select_actions(pred,causal["allowed"],arm="physical_neural");np.testing.assert_array_equal(choices,actual)
            policies[arm]=prior.selected_paths(paths,choices,row["baseline_index"])
        for site in sorted(set(data["sites"])):
            at=data["sites"]==site
            scored={k:prior.metrics.policy(v[at],floor[at],paths[at,1],target[at],valid[at],data["recordings"][at],
                easy_cut=row["easy_cut"],hard_cut=row["hard_cut"]) for k,v in policies.items()}
            old=json.loads((parent.PUBLIC/"groups"/(row["trial"]+"_"+str(site)+".json")).read_text())
            if scored["disabled"]!=old["policies"]["mean_absolute"]:raise ValueError("Original disabled control scoring changed")
            effects={"real_minus_"+a:prior.metrics.paired(policies[a][at],policies["real"][at],floor[at],target[at],valid[at]) for a in ("disabled","noise")}
            value=dict(trial=row["trial"],seed=row["seed"],fold=row["fold"],site=str(site),policies=scored,effects=effects,
                registration_sha256=sha(REG),policy_freeze_sha256=sha(PUBLIC/"policy_freeze.json"))
            p=PUBLIC/"groups"/(row["trial"]+"_"+str(site)+".json")
            if verify or resume and p.exists():
                if json.loads(p.read_text())!=value:raise ValueError("Metric group replay differs")
            else:once(p,value)
            refs.append(artifact(p));rows.append(value)
        beat(state="readout_group_complete",trial=row["trial"],groups=len(rows),total=54)
    s=summarize(rows,cfg)
    if verify:
        if json.loads((PUBLIC/"summary.json").read_text())!=s:raise ValueError("Full summary replay differs")
        once(finish,dict(**receipt(),all27_new_inferences_exact=True,all54_metric_groups_exact=True,
            original_disabled_control_exact=True,summary_sha256=sha(PUBLIC/"summary.json"),complete_sha256=sha(PUBLIC/"complete.json")))
    else:
        once(PUBLIC/"summary.json",s);once(finish,dict(**receipt(),groups=refs,summary_sha256=sha(PUBLIC/"summary.json")))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("phase",choices=("register","features","feature_verify","pilot","train","infer","eval","verify"))
    parser.add_argument("--resume",action="store_true");args=parser.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=="register":admission(True)
        elif args.phase in ("features","feature_verify"):build_features(args.resume,args.phase=="feature_verify")
        elif args.phase=="pilot":pilot()
        elif args.phase=="train":train(args.resume)
        elif args.phase=="infer":infer(args.resume)
        else:evaluate(args.phase=="verify",args.resume)
