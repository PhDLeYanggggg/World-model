"""Source-only cutoff-context repair with frozen forecasts and risk definition."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
import time
if platform.system() == "Darwin" and platform.machine() != "arm64":
    raise RuntimeError("Native arm64 .venv-pytorch required")
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import torch
from scripts import run_m3w_dimensionless_risk as base
from src.world_model.m3w_cutoff_relative_risk import features, information_loss_witness

CONFIG = "configs/m3w_cutoff_relative_risk_v1.json"
PARENT = ROOT/"data/stage_cvpr2027_experiments/dimensionless_risk_v1"
PARENT_REPORT = ROOT/"outputs/publication_readiness_2026_09/dimensionless_risk_v1/analysis.json"
CODE = ("scripts/run_m3w_cutoff_relative_risk.py", "src/world_model/m3w_cutoff_relative_risk.py",
        "tests/test_m3w_cutoff_relative_risk.py")


def read(path):
    return json.loads(Path(path).read_text())


def load():
    cfg = read(ROOT/CONFIG)
    assert base.file_digest(PARENT/"identity.json") == cfg["parent_identity_sha256"]
    assert base.file_digest(PARENT_REPORT) == cfg["parent_analysis_sha256"]
    pcfg, pack, refs, pi = base.load()
    assert pi == read(PARENT/"identity.json")
    for k in ("sites", "seeds", "actions", "forest", "easy_rho", "bootstrap_resamples"):
        assert cfg[k] == pcfg[k]
    assert cfg["policies"] == ["cutoff_point", "cutoff_population", "cutoff_selected"]
    assert len(pcfg["policies"]) == 10
    assert not any(cfg[k] for k in ("new_predictor_training", "threshold_search", "external_readout",
                                  "deployment", "stage5c_executed", "smc_enabled"))
    controls = base.receipts(pcfg, PARENT, pack[1][4])
    bindings = dict(pi["source_bindings"])
    for r in controls.values(): bindings[r["checkpoint"]] = r["checkpoint_sha256"]
    for path in (ROOT/CONFIG, ROOT/cfg["registration"], PARENT/"identity.json", PARENT_REPORT,
                 *(ROOT/p for p in CODE)):
        bindings[str(path.relative_to(ROOT))] = base.file_digest(path)
    for action in cfg["actions"]:
        for seed in cfg["seeds"]:
            path = PARENT/"outcomes"/f"{action}_seed{seed}.npz"
            bindings[str(path.relative_to(ROOT))] = base.file_digest(path)
    identity = dict(config=cfg, source_bindings=bindings, targets=list(base.TARGETS),
        runtime=pi["runtime"], numpy=np.__version__, torch=torch.__version__,
        sklearn=__import__("sklearn").__version__, scope="design_exposed_source_only_not_confirmation")
    base.assert_current(identity)
    return dict(cfg=cfg, pcfg=pcfg, pack=pack, refs=refs, controls=controls, identity=identity)


def train(ctx, root, args, beat):
    cfg, pack = ctx["cfg"], ctx["pack"]
    for key in pack[1][4]:
        if args.view and key != args.view: continue
        for action in cfg["actions"]:
            if args.action and action != args.action: continue
            ids, raw, d, pr, q, draws, s = base.prepare(pack, key, action, ctx["refs"])
            cutoff = pack[2][5][key]["positive_easy_cut"]
            x = features(raw, d, s, cutoff); pp = base.preprocess(x, pr)
            reference = ctx["controls"][key, action, "native"]
            ti = dict(experiment_sha256=base.file_digest(root/"identity.json"), view=key, action=action,
                feature_width=x.shape[1], cutoff=cutoff, inputs_sha256=base.array_hash(ids,x,d),
                targets_sha256=base.array_hash(q), draws_sha256=base.array_hash(draws),
                known_sha256=base.array_hash(pr["known"]), training_sites=pr["training_sites"],
                parent_native_fit_sha256=reference["checkpoint_sha256"])
            for k in ("targets_sha256", "draws_sha256", "known_sha256", "training_sites"):
                assert ti[k] == reference["identity"][k]
            folder = root/"trials"/key/action; receipt = folder/"complete.json"
            if receipt.exists():
                r = read(receipt)
                assert r["identity"] == ti and r["fit"]["complete"]
                assert base.file_digest(ROOT/r["checkpoint"]) == r["checkpoint_sha256"]
                beat(state="cached_verified_fit",view=key,action=action); continue
            result = base.fit(x,q,d,pp,draws,settings=cfg["forest"],seed=int(key.rsplit("_seed",1)[1]),
                identity=ti,directory=folder,resume=args.resume,stop_at=args.stop_at,
                heartbeat=lambda **v: beat(view=key,action=action,**v))
            if not result["complete"]:
                beat(state="pilot_checkpoint_not_full_matrix",view=key,action=action); return False
            cp = folder/"checkpoint.joblib"
            base.immutable_json(receipt,dict(identity=ti,fit=result,checkpoint=str(cp.relative_to(ROOT)),
                checkpoint_sha256=base.file_digest(cp),result_source="fresh_run"))
    return True


def receipts(ctx, root):
    records = {}
    for key in ctx["pack"][1][4]:
        for action in ctx["cfg"]["actions"]:
            r = read(root/"trials"/key/action/"complete.json")
            assert r["identity"]["experiment_sha256"] == base.file_digest(root/"identity.json")
            assert r["identity"]["view"] == key and r["identity"]["action"] == action
            assert key.rsplit("_seed",1)[0] not in r["identity"]["training_sites"]
            assert r["fit"]["complete"] and r["fit"]["trees"] == 128
            assert base.file_digest(ROOT/r["checkpoint"]) == r["checkpoint_sha256"]
            records[key,action] = r
    assert len(records) == 36
    return records


def decide(ctx, root, args, beat):
    cfg, ap = ctx["cfg"], ctx["pack"][1]; data = ap[1]
    fitted = receipts(ctx,root); archives = []; count = 0
    for key in ap[4]:
        for action in cfg["actions"]:
            values,p,_,cutoff = base.causal_view(ap,key,action)
            ids = values["ids"]; g,s = data["geometry"][ids],data["scale"][ids]
            raw,d,_ = base.native_features(g,p,s)
            r = fitted[key,action]; cp = joblib.load(ROOT/r["checkpoint"])
            assert cp["identity"] == r["identity"] and cp["settings"] == cfg["forest"]
            assert cp["seed"] == int(key.rsplit("_seed",1)[1])
            assert r["identity"]["cutoff"] == cutoff
            x = features(raw,d,s,cutoff); f = base.predict(cp["model"],x,cp["preprocess"])
            probes = []
            for factor in (.01,100.):
                xx = features(raw,d*factor,s*factor,cutoff*factor)
                ff = base.predict(cp["model"],xx,cp["preprocess"])
                probes.append(dict(factor=factor,feature_changed_rows=int(np.any(xx!=x,axis=1).sum()),
                    prediction_changed_rows=int(np.any(ff!=f,axis=1).sum()),
                    max_feature_difference=float(np.abs(xx-x).max()),max_prediction_difference=float(np.abs(ff-f).max())))
            bank = base.scores(f,d,cutoff,g[:,:16].reshape(-1,8,2),cfg["easy_rho"])
            bits = np.zeros((len(ids),3),bool); bits[:,0] = bank["point"]
            index = base.scene_index(data["recordings"][ids],data["frames"][ids],data["tracks"][ids])
            reports = []
            for j in range(len(index["offsets"])-1):
                q = index["order"][index["offsets"][j]:index["offsets"][j+1]]
                gain,risk,den,ok = (bank[k][q] for k in ("gain","risk","denominator","eligible"))
                v,pop = base.allocate(gain,risk,ok,cfg["easy_rho"]*float(den.sum()),seconds=cfg["solver_seconds"])
                bits[q,1] = v
                v,sel = base.allocate(gain,np.maximum(risk,0)-cfg["easy_rho"]*den,ok,0.,seconds=cfg["solver_seconds"])
                bits[q,2] = v
                reports.append(dict(recording=str(index["recordings"][j]),frame=int(index["frames"][j]),
                    population=pop,selected=sel))
            path = root/"decisions"/f"{key}_{action}.npz"
            if args.verify and not path.exists(): raise ValueError("Replay cannot create missing decisions")
            base.write_arrays(path,dict(ids=ids,choices=bits,distance=d,fractions=f))
            rp = path.with_suffix(".json")
            base.immutable_json(rp,dict(experiment_sha256=base.file_digest(root/"identity.json"),
                view=key,action=action,cutoff=cutoff,path=str(path.relative_to(ROOT)),sha256=base.file_digest(path),
                probes=probes,queries=reports))
            archives.append(dict(path=str(rp.relative_to(ROOT)),sha256=base.file_digest(rp)))
            count += len(reports); beat(state="decisions_saved",view=key,action=action,instances=count)
    assert count == 188388
    base.immutable_json(root/"decisions_complete.json",dict(experiment_sha256=base.file_digest(root/"identity.json"),
        archives=archives,queries=count,fits=36,outcomes_used_for_choices=False))
    if args.verify: base.immutable_json(ROOT/cfg["reports"]/"decision_replay.json",dict(exact_arrays=True,exact_reports=True))


def evaluate(ctx, root, args, beat):
    cfg,ap = ctx["cfg"],ctx["pack"][1]; data = ap[1]; n = len(data["sites"])
    fitted = receipts(ctx,root); done = read(root/"decisions_complete.json")
    assert done["experiment_sha256"] == base.file_digest(root/"identity.json")
    decisions,diagnostics,probes = {},[],[]
    for ref in done["archives"]:
        assert base.file_digest(ROOT/ref["path"]) == ref["sha256"]
        r = read(ROOT/ref["path"]); assert base.file_digest(ROOT/r["path"]) == r["sha256"]
        decisions[r["view"],r["action"]] = r
        diagnostics.extend(q[k] for q in r["queries"] for k in ("population","selected"))
        probes.extend(dict(view=r["view"],action=r["action"],**p) for p in r["probes"])
    old = read(PARENT_REPORT); policies = ctx["pcfg"]["policies"]+cfg["policies"]
    summary,contrasts,quality = {},{},[]
    for action in cfg["actions"]:
        errors = {p:[] for p in policies}; ends = {p:[] for p in policies}; detail = {p:{} for p in policies}
        for seed in cfg["seeds"]:
            ref = next(r for r in ap[3]["outcome_archives"] if r["path"].endswith(f"{action}_seed{seed}.npz"))
            assert base.file_digest(ROOT/ref["path"]) == ref["sha256"]
            with np.load(ROOT/ref["path"],allow_pickle=False) as z: o = {k:z[k].copy() for k in z.files}
            cv,cf = o["cv"],o["cf"]; masks = {k:o[k] for k in ("complete","zero_CV","positive_easy","hard")}
            oldpath = PARENT/"outcomes"/f"{action}_seed{seed}.npz"
            assert base.file_digest(oldpath) == ctx["identity"]["source_bindings"][str(oldpath.relative_to(ROOT))]
            choices = np.zeros((n,len(policies)),bool); seen = np.zeros(n,bool)
            with np.load(oldpath,allow_pickle=False) as z: choices[:,:10] = z["choices"]
            for key in ap[4]:
                if not key.endswith(f"seed{seed}"): continue
                r = decisions[key,action]
                with np.load(ROOT/r["path"],allow_pickle=False) as z:
                    ids = z["ids"].copy(); choices[ids,10:] = z["choices"]
                    assert not seen[ids].any(); seen[ids] = True
                    costs,reference = base.supported_costs(o["candidate_ade"][ids],cv[ids],masks["complete"][ids])
                    q = base.targets(costs,reference,z["distance"],masks["complete"][ids],r["cutoff"])
                    supported = masks["complete"][ids]
                    quality.append(dict(view=key,action=action,complete_rows=int(supported.sum()),
                        held_source_fraction_mse=((z["fractions"][supported]-q[supported])**2).mean(0).tolist()))
            assert seen.all()
            def metric(e,reference=cv,mask=None,ci=False):
                use = np.ones(n,bool) if mask is None else mask
                return base.paired_scene_metrics(e[use],reference[use],data["sites"][use],expected_scenes=cfg["sites"],
                    dataset="sdd",coordinate_unit="annotation_pixel",bootstrap_resamples=3000 if ci else 0)
            for j,p in enumerate(policies):
                bits = choices[:,j]; e = np.where(bits,o["candidate_ade"],cv); f = np.where(bits,o["candidate_fde"],cf)
                errors[p].append(e); ends[p].append(f)
                detail[p][str(seed)] = dict(ADE=metric(e),FDE=metric(f,cf),selected=int(bits.sum()),
                    subsets={k:metric(e,mask=m) for k,m in masks.items()},
                    selected_unknown=int((bits&np.isnan(cv)).sum()),selected_incomplete=int((bits&~masks["complete"]).sum()),
                    zero_CV_harmed=int((bits&masks["zero_CV"]&(e>0)).sum()),
                    full_grid_gain_bounds={site:[float(np.where(bits,o[b],0)[data["sites"]==site].mean()) for b in ("lower","upper")] for site in cfg["sites"]})
            base.write_arrays(root/"outcomes"/f"{action}_seed{seed}.npz",dict(choices=choices[:,10:]))
        mean = {p:np.mean(errors[p],axis=0) for p in policies}
        for p in policies:
            row = dict(ADE=metric(mean[p],ci=True),FDE=metric(np.mean(ends[p],axis=0),cf,ci=True),
                subsets={k:metric(mean[p],mask=m,ci=True) for k,m in masks.items()},seeds=detail[p])
            summary[action+"__"+p] = row
            if p in ctx["pcfg"]["policies"]: assert row == old["summary"][action+"__"+p]
        for pol in ("point","population","selected"):
            left = "cutoff_"+pol
            for right in ("native_"+pol,"dimensionless_"+pol,"old_strict"):
                contrast = {}
                for subset,mask in (("all",None),("hard",masks["hard"]),("positive_easy",masks["positive_easy"])):
                    l,r = metric(mean[left],mask=mask)["by_scene"],metric(mean[right],mask=mask)["by_scene"]
                    contrast[subset] = base.paired_scene_contrast([l[s]["gain_percent"] for s in cfg["sites"]],
                        [r[s]["gain_percent"] for s in cfg["sites"]],resamples=3000)
                contrasts[action+"__"+left+"_minus_"+right] = contrast
        beat(state="aggregate_complete",action=action)
    public = ROOT/cfg["reports"]
    result = dict(result_sources=dict(new_head_training="fresh_run",old_controls_predictors_targets="cached_verified",
        decisions_and_reductions="fresh_run",external_readout="not_run",independent_calibration="not_run"),
        experiment_sha256=base.file_digest(root/"identity.json"),decisions_sha256=base.file_digest(root/"decisions_complete.json"),
        parent_analysis_sha256=cfg["parent_analysis_sha256"],rows=n,sites=cfg["sites"],seeds=cfg["seeds"],policies=policies,
        summary=summary,contrasts=contrasts,held_source_fit_quality=quality,unit_probes=probes,
        metadata_unit_predictions_exact=all(p["prediction_changed_rows"]==0 for p in probes),
        synthetic_information_loss_witness=information_loss_witness(),old_summary_rows_exact=30,
        solver=dict(instances=len(diagnostics),optimal_unverified=sum(not r["optimal"] for r in diagnostics),
            risk_violations=sum(not r["constraint_pass"] for r in diagnostics)),
        fits=[dict(view=k,action=a,**r) for (k,a),r in fitted.items()],
        scope="design_exposed_source_only_not_confirmation",deployment=False,stage5c_executed=False,smc_enabled=False)
    if args.verify and not (public/"analysis.json").exists(): raise ValueError("Missing original aggregate")
    base.immutable_json(public/"analysis.json",result)
    if args.verify: base.immutable_json(public/"aggregate_replay.json",dict(exact=True,analysis_sha256=base.file_digest(public/"analysis.json")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase",choices=["preflight","train","decide","evaluate","all"],default="preflight")
    p.add_argument("--resume",action="store_true"); p.add_argument("--verify",action="store_true")
    p.add_argument("--view"); p.add_argument("--action"); p.add_argument("--stop-at",type=int)
    args = p.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg = read(ROOT/CONFIG); root = ROOT/cfg["output"]; root.mkdir(parents=True,exist_ok=True)
    def beat(**values):
        event = dict(pid=os.getpid(),updated_unix=time.time(),**values)
        base.json_write(root/"heartbeat.json",event)
        with (root/"events.jsonl").open("a") as f: f.write(json.dumps(event)+"\n")
        print(json.dumps(event),flush=True)
    with (root/"runner.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        beat(state="verifying_source_dependencies")
        ctx = load(); base.immutable_json(root/"identity.json",ctx["identity"])
        if args.view and args.view not in ctx["pack"][1][4]: raise ValueError("Unknown view")
        if args.action and args.action not in cfg["actions"]: raise ValueError("Unknown action")
        beat(state="preflight_complete",rows=len(ctx["pack"][1][1]["sites"]))
        if args.phase in ("train","all") and not train(ctx,root,args,beat): return
        if args.phase in ("decide","all"): decide(ctx,root,args,beat)
        if args.phase in ("evaluate","all"): evaluate(ctx,root,args,beat)
        base.assert_current(ctx["identity"]); beat(state="phase_complete",phase=args.phase)


if __name__ == "__main__": main()
