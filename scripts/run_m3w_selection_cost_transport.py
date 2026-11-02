"""Read-only SOURCE fitting versus exposed-selection risk transport diagnosis."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time

if platform.system()=="Darwin" and platform.machine()!="arm64": raise RuntimeError("Native arm64 before Torch")
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS"): os.environ.setdefault(key,"4")
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_selection_controller as prior
from src.evaluation import m3w_selection_cost_transport as api
from src.evaluation.m3w_selected_cost_diagnosis import MOMENTS

NAME="european_selection_cost_transport_v1"
PUBLIC=ROOT/"outputs/publication_readiness_2026_09"/NAME
PRIVATE=ROOT/"data/stage_cvpr2027_experiments"/NAME
CONFIG=ROOT/"configs/m3w_selection_cost_transport_v1.json";REG=PUBLIC/"registration_v2.json"
sha,artifact,checked,write,once=prior.sha,prior.artifact,prior.checked,prior.write,prior.once
FILES=(Path(__file__),CONFIG,ROOT/"src/evaluation/m3w_selection_cost_transport.py",
       ROOT/"tests/test_m3w_selection_cost_transport.py",PUBLIC/"protocol.md")


def admission(register=False):
    cfg=json.loads(CONFIG.read_text());_,producers,whitelist=prior.admission();models=prior.fitted()
    verification=json.loads((prior.PUBLIC/"verification.json").read_text())
    if (not verification["all18_cost_inference_exact"] or not verification["all54_frozen_action_metric_groups_exact"]
            or verification["summary_sha256"]!=sha(prior.PUBLIC/"summary.json")):
        raise ValueError("Complete frozen critic/policy verification required")
    if cfg["expected_groups"]!=252 or cfg["positive_harm_budget"]!=.02 or any(cfg[k] for k in
        ("new_training","threshold_search","calibration_fitted","normalization_refitted","policy_changed",
         "calibration_access","confirmation_access","stage5c_executed","smc_enabled")):
        raise ValueError("Read-only252group diagnosis with unchanged decisions required")
    dependencies=("src/evaluation/m3w_selected_cost_diagnosis.py","src/world_model/m3w_selection_controller.py",
                  "src/world_model/m3w_multiaction_cost.py","scripts/run_m3w_selection_controller.py")
    parent_paths=(prior.REG,prior.PUBLIC/"training_freeze.json",prior.PUBLIC/"source_bank_freeze.json",
                  prior.PUBLIC/"policy_freeze.json",prior.PUBLIC/"complete.json",prior.PUBLIC/"verification.json",
                  prior.OUTCOMES/"complete.json")
    value=dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        prior_pre_execution_draft_registration=artifact(PUBLIC/"registration.json"),
        pre_execution_fixture_correction="synthetic_guard_rejection_fixture_needs_positive_not_zero_predicted_gain_no_diagnostic_data_or_method_change",
        dependencies={p:sha(ROOT/p) for p in dependencies},parents=[artifact(p) for p in parent_paths],
        models=18,source_groups=144,selection_groups=108,total_groups=252,source_resubstitution_not_validation=True,
        selection_already_exposed=True,normalization_refitted=False,policy_changed=False,threshold_search=False,
        new_training=False,calibration_access=False,confirmation_access=False,stage5c_executed=False,smc_enabled=False)
    if register:once(REG,value)
    if json.loads(REG.read_text())!=value:raise ValueError("Diagnostic definition changed")
    if not register:
        for p in (*FILES,REG):
            blob=subprocess.run(["git","show","HEAD:"+str(p.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
            if hashlib.sha256(blob).hexdigest()!=sha(p):raise ValueError("Commit the diagnostic before execution")
    return cfg,producers,whitelist,models


def runtime(phase):
    torch.set_num_threads(4);torch.set_num_interop_threads(1);start=time.monotonic()
    def guard():
        if shutil.disk_usage(ROOT).free<10*2**30:raise OSError("Keep10GiB free")
        if time.monotonic()-start>43200:raise TimeoutError("12hboundary;resume unchanged diagnosis")
    def beat(**kw):
        guard();v=dict(pid=os.getpid(),utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),phase=phase,**kw)
        write(PRIVATE/"heartbeat.json",v);print(json.dumps(v),flush=True)
    def receipt():return dict(pid=os.getpid(),seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,registration_sha256=sha(REG))
    beat(state="runtime_ready",architecture=platform.machine(),device="cpu",threads=4,workers=0,new_training=False)
    return beat,guard,receipt


def source_case(row,source,whitelist):
    ids=np.asarray(whitelist[~np.isin(source["sites"][whitelist],row["fit_sites"])])
    ref=json.loads((prior.PRIVATE/"source_banks"/row["trial"]/"complete.json").read_text())
    if ref["ids_sha256"]!=prior.frozen.fingerprint(ids):raise ValueError("Original SOURCE cost row scope required")
    neural=np.load(checked(ref["prediction"]),mmap_mode="r",allow_pickle=False);origin=source["origin"][ids]
    physical=np.stack([prior.baseline_numpy(source["history"][ids],k)-origin[:,None] for k in range(6)],1)
    paths=np.concatenate((physical,neural[:,None].astype(float)),1)
    causal=prior.api.features(geometry=source["geometry"][ids],detector_history=source["detector_history"][ids],
        rollouts=paths,floor_index=row["baseline_index"],recordings=source["recordings"][ids],frames=source["frames"][ids],
        ids=ids,sites=source["sites"][ids])
    # Native labels are recomputed directly, never restored from float32 training targets.
    target=np.load(prior.SOURCE/"target_eval.npy",mmap_mode="r",allow_pickle=False)[ids]-origin[:,None]
    valid=np.load(prior.SOURCE/"valid.npy",mmap_mode="r",allow_pickle=False)[ids]
    y,_=prior.api.labels(paths,target,valid,row["baseline_index"],row["easy_cut"])
    return dict(causal=causal,target=y,recordings=source["recordings"][ids])


def selection_case(row):
    data,paths,causal=prior.selection_bank(row);target,valid=prior.selection_labels(data)
    native,_=prior.api.labels(paths,target,valid,row["baseline_index"],row["easy_cut"])
    return dict(causal=causal,target=native,recordings=data["recordings"])


def pooled(rows):
    scopes=("all_legal","gate_passing","selected","neural_legal")
    result={}
    ratio=lambda a,b:float(a/b) if b>0 else None
    for domain in ("source_fit","selection"):
        result[domain]={}
        for arm in ("row","context"):
            cells=[r for r in rows if r["domain"]==domain and r["arm"]==arm];groups={}
            for scope in scopes:
                values=[r["diagnosis"][scope] for r in cells]
                p=np.array([[v["predicted_known_mass"][k] for k in MOMENTS] for v in values]).sum(0)
                y=np.array([[v["observed_known_mass"][k] for k in MOMENTS] for v in values]).sum(0)
                records=[v for r in cells for v in r["selected_recordings"].values()] if scope=="selected" else []
                groups[scope]=dict(known_action_views=sum(v["known_action_views"] for v in values),
                    unknown_action_views=sum(v["unknown_action_views"] for v in values),
                    predicted_known_mass=dict(zip(MOMENTS,p.tolist())),observed_known_mass=dict(zip(MOMENTS,y.tolist())),
                    predicted_all_risk_on_known=ratio(p[1],p[2]),observed_all_risk_on_known=ratio(y[1],y[2]),
                    predicted_easy_risk_on_known=ratio(p[4],p[3]),observed_easy_risk_on_known=ratio(y[4],y[3]),
                    reference_inflation_ratio=ratio(p[2],y[2]),easy_reference_inflation_ratio=ratio(p[3],y[3]),
                    harm_prediction_ratio=ratio(p[1],y[1]),easy_harm_prediction_ratio=ratio(p[4],y[4]),
                    harm_gap=float(y[1]-p[1]),all_reference_gap=float(.02*(p[2]-y[2])),
                    easy_harm_gap=float(y[4]-p[4]),easy_reference_gap=float(.02*(p[3]-y[3])),
                    selected_known_all_violating_record_views=sum(v["observed_all_risk_on_known"] is not None and v["observed_all_risk_on_known"]>.02 for v in records),
                    selected_known_easy_violating_record_views=sum(v["observed_easy_risk_on_known"] is not None and v["observed_easy_risk_on_known"]>.02 for v in records),
                    selected_unknown_or_empty_easy_record_views=sum(v["observed_easy_risk_on_known"] is None for v in records))
            funnel={name:dict(rows=sum(r["diagnosis"]["neural_funnel"]["buckets"][name]["rows"] for r in cells),
                known_benefit_mass=sum(r["diagnosis"]["neural_funnel"]["buckets"][name]["known_benefit_mass"] for r in cells))
                for name in cells[0]["diagnosis"]["neural_funnel"]["buckets"]}
            result[domain][arm]=dict(groups=len(cells),subsets=groups,neural_funnel=funnel,
                useful_known_neural_rows=sum(r["diagnosis"]["neural_funnel"]["useful_known_neural_rows"] for r in cells),
                useful_known_neural_benefit_mass=sum(r["diagnosis"]["neural_funnel"]["useful_known_neural_benefit_mass"] for r in cells))
    return dict(groups=len(rows),pooled_descriptive_action_view_diagnostics=result,
        repeated_action_views_not_independent=True,source_resubstitution_not_validation=True,
        new_policy_evidence=False,calibration_fitted=False,policy_changed=False,threshold_search=False,
        calibration_access=False,confirmation_access=False,deployment_changed=False,submission_ready=False,
        stage5c_executed=False,smc_enabled=False)


def execute(phase,resume=False):
    cfg,producers,whitelist,models=admission();beat,guard,receipt=runtime(phase);verify=phase=="verify";pilot=phase=="pilot"
    finish=PUBLIC/("verification.json" if verify else "pilot.json" if pilot else "complete.json")
    if finish.exists():raise ValueError("Complete diagnosis phase immutable")
    if not pilot and not json.loads((PUBLIC/"pilot.json").read_text())["engineering_pass"]:
        raise ValueError("Actual first-producer resource pilot required")
    source=prior.source_inputs(json.loads(prior.frozen.REG.read_text()));refs=[];all_rows=[];cases=[]
    for row in producers[:1] if pilot else producers:
        for domain,build in (("source_fit",lambda:source_case(row,source,whitelist)),("selection",lambda:selection_case(row))):
            for arm in ("row","context"):
                casefile=PUBLIC/"cases"/(row["trial"]+"_"+domain+"_"+arm+".json")
                if casefile.exists() and resume and not verify:
                    case=json.loads(casefile.read_text())
                    if case["registration_sha256"]!=sha(REG):raise ValueError("Diagnostic checkpoint binding changed")
                    for ref in case["groups"]:
                        value=json.loads(checked(ref).read_text());all_rows.append(value);refs.append(ref)
                    cases.append(artifact(casefile));beat(state="case_cached_verified",cases=len(cases),total=36);continue
                guard();data=build();c=data["causal"];state=prior.api.read_checkpoint(checked(models[row["trial"],arm]["checkpoint"]))
                if domain=="source_fit":
                    np.testing.assert_array_equal(state["fit_ids"],c["ids"])
                    prediction=prior.api.predict(state,c)
                    choices,_=prior.allocator.select_actions(prediction,c["allowed"],arm="physical_neural")
                    joint=None
                else:
                    ref=json.loads((prior.PUBLIC/"policies"/(row["trial"]+"_"+arm+".json")).read_text())
                    prediction=np.load(checked(ref["prediction"]),mmap_mode="r",allow_pickle=False)
                    choices_saved=np.load(checked(ref["choices"]),allow_pickle=False)
                    choices,_=prior.allocator.select_actions(prediction,c["allowed"],arm="physical_neural")
                    np.testing.assert_array_equal(choices,choices_saved[ref["policy_names"].index("independent")])
                    joint=choices_saved[ref["policy_names"].index("joint_matched")]
                case_refs=[]
                for site in sorted(set(c["sites"])):
                    at=np.flatnonzero(c["sites"]==site)
                    diagnostic=api.diagnose(prediction[at],data["target"][at],c["allowed"][at],choices[at],c["scales"][at])
                    records={str(rec):api.selected_summary(prediction[ix],data["target"][ix],c["allowed"][ix],choices[ix],c["scales"][ix])
                        for rec in sorted(set(data["recordings"][at])) for ix in [at[data["recordings"][at]==rec]]}
                    groups=[g for g in c["groups"] if c["sites"][g[0]]==site]
                    value=dict(trial=row["trial"],seed=row["seed"],source_fold=row["fold"],domain=domain,arm=arm,site=str(site),
                        diagnosis=diagnostic,selected_recordings=records,
                        input_statistics=api.feature_shift(c["x"][at],state["preprocess"],groups),
                        prediction_sha256=prior.frozen.fingerprint(prediction[at]),target_cost_sha256=prior.frozen.fingerprint(data["target"][at]),
                        choices_sha256=prior.frozen.fingerprint(choices[at]),registration_sha256=sha(REG))
                    if joint is not None:value["joint_selected"]=api.selected_summary(prediction[at],data["target"][at],c["allowed"][at],joint[at],c["scales"][at])
                    path=PUBLIC/"groups"/(row["trial"]+"_"+domain+"_"+arm+"_"+str(site)+".json")
                    if verify or path.exists() and resume:
                        if json.loads(path.read_text())!=value:raise ValueError("Diagnostic inference or scalar replay differs")
                    else:once(path,value)
                    all_rows.append(value);refs.append(artifact(path));case_refs.append(artifact(path))
                case=dict(trial=row["trial"],domain=domain,arm=arm,groups=case_refs,
                    full_prediction_sha256=prior.frozen.fingerprint(prediction),rows=len(c["ids"]),registration_sha256=sha(REG))
                if verify:
                    if json.loads(casefile.read_text())!=case:raise ValueError("Complete SOURCE inference replay differs")
                else:once(casefile,case)
                cases.append(artifact(casefile));beat(state="case_complete",cases=len(cases),total=4 if pilot else 36)
                del data,state,prediction
    if pilot:
        if len(all_rows)!=28 or len(cases)!=4:raise ValueError("Complete first-producer diagnostic pilot required")
        actual=receipt()
        once(finish,dict(**actual,engineering_pass=True,cases=cases,groups=28,
            projected_all36_case_seconds_not_completion=actual["seconds"]*9,
            new_training=False,policy_changed=False))
    else:
        if len(all_rows)!=252 or len(cases)!=36:raise ValueError("All144SOURCE and108selection groups required")
        summary=pooled(all_rows)
        if verify:
            if json.loads((PUBLIC/"summary.json").read_text())!=summary:raise ValueError("Full pooled scalar summary differs")
            once(finish,dict(**receipt(),all18_SOURCE_cost_inferences_exact=True,all252_diagnostic_groups_exact=True,
                selection_predictions_cached_verified=True,new_optimization=False,
                summary_sha256=sha(PUBLIC/"summary.json"),complete_sha256=sha(PUBLIC/"complete.json")))
        else:
            once(PUBLIC/"summary.json",summary);once(finish,dict(**receipt(),cases=cases,groups=refs,
                summary_sha256=sha(PUBLIC/"summary.json"),new_training=False,policy_changed=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("phase",choices=("register","pilot","run","verify"));parser.add_argument("--resume",action="store_true");args=parser.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=="register":admission(True)
        else:execute(args.phase,args.resume)
