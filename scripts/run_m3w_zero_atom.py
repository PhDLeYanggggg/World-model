"""Source-only zero-reference leaf fits and protected, matched-count readout."""
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
from scripts import run_m3w_cutoff_relative_risk as parent
from src.world_model.m3w_native_gain_harm import standardized
from src.world_model import m3w_zero_atom as atom

base = parent.base
CONFIG = "configs/m3w_zero_atom_v1.json"
CODE = ("scripts/run_m3w_zero_atom.py", "src/world_model/m3w_zero_atom.py", "tests/test_m3w_zero_atom.py")


def read(path):
    return json.loads(Path(path).read_text())


def load():
    cfg = read(ROOT/CONFIG); ctx = parent.load(); pr = ROOT/ctx["cfg"]["output"]
    assert ctx["identity"] == read(ROOT/cfg["parent_identity"])
    assert base.file_digest(ROOT/cfg["parent_analysis"]) == cfg["parent_analysis_sha256"]
    assert all(cfg[k] == ctx["cfg"][k] for k in ("sites", "seeds", "actions", "easy_rho", "bootstrap_resamples"))
    assert cfg["rules"] == ["point", "population", "selected"] and cfg["zero_reference_tolerance"] == 0
    assert not any(cfg[k] for k in ("new_forecast_training", "new_forest_training", "threshold_search", "external_readout", "deployment", "stage5c_executed", "smc_enabled"))
    refs = parent.receipts(ctx, pr); old = read(ROOT/cfg["parent_analysis"])
    bindings = dict(ctx["identity"]["source_bindings"])
    for r in refs.values(): bindings[r["checkpoint"]] = r["checkpoint_sha256"]
    for p in (CONFIG, cfg["parent_identity"], cfg["parent_analysis"], cfg["registration"], *CODE):
        bindings[p] = base.file_digest(ROOT/p)
    path = pr/"decisions_complete.json"
    assert base.file_digest(path) == old["decisions_sha256"]
    bindings[str(path.relative_to(ROOT))] = base.file_digest(path)
    decisions = {}
    for ref in read(path)["archives"]:
        assert base.file_digest(ROOT/ref["path"]) == ref["sha256"]
        r = read(ROOT/ref["path"]); assert base.file_digest(ROOT/r["path"]) == r["sha256"]
        decisions[r["view"], r["action"]] = r
        bindings[ref["path"]], bindings[r["path"]] = ref["sha256"], r["sha256"]
    identity = dict(config=cfg, source_bindings=bindings, runtime=ctx["identity"]["runtime"],
        role="design_exposed_source_only_not_confirmation", sklearn=ctx["identity"]["sklearn"])
    base.assert_current(identity)
    return dict(cfg=cfg, parent=ctx, refs=refs, decisions=decisions, old=old, identity=identity)


def train(ctx, root, args, beat):
    pc, cfg = ctx["parent"], ctx["cfg"]
    data = pc["pack"][1][1]
    for key in pc["pack"][1][4]:
        if args.view and key != args.view: continue
        for action in cfg["actions"]:
            if args.action and action != args.action: continue
            ids, raw, d, pr, q, draws, scale = base.prepare(pc["pack"], key, action, pc["refs"])
            ref = ctx["refs"][key, action]; frozen = joblib.load(ROOT/ref["checkpoint"])
            assert frozen["identity"] == ref["identity"]
            cutoff = ref["identity"]["cutoff"]
            x = parent.features(raw, d, scale, cutoff)
            assert base.array_hash(ids, x, d) == ref["identity"]["inputs_sha256"]
            assert base.array_hash(q) == ref["identity"]["targets_sha256"]
            np.testing.assert_array_equal(draws, frozen["draws"])
            event = atom.zero_event(q, pr["known"]); w = atom.source_weights(pr["known"], draws, d)
            np.testing.assert_array_equal(w, frozen["sample_weight"])
            use = w > 0; z = standardized(x, frozen["preprocess"])
            np.testing.assert_allclose(q[event & use, 2], 1., atol=2e-6)
            assert not q[event & use, 0].any()
            g = data["geometry"][ids, :16].reshape(-1, 8, 2)
            moving = np.any(g[:, -1] != g[:, -2], axis=1)
            ti = dict(experiment_sha256=base.file_digest(root/"identity.json"), view=key, action=action,
                forest_sha256=ref["checkpoint_sha256"], event_sha256=base.array_hash(event),
                weight_sha256=base.array_hash(w), training_sites=pr["training_sites"],
                input_sha256=ref["identity"]["inputs_sha256"], targets_sha256=ref["identity"]["targets_sha256"])
            folder = root/"readouts"/key/action; folder.mkdir(parents=True, exist_ok=True)
            path, receipt = folder/"checkpoint.joblib", folder/"complete.json"
            if receipt.exists():
                r = read(receipt); assert r["identity"] == ti and base.file_digest(path) == r["checkpoint_sha256"]
                beat(state="cached_verified_readout", view=key, action=action); continue
            entries, seconds = [], 0.
            if path.exists():
                if not args.resume: raise ValueError("Existing readout requires --resume")
                cp = joblib.load(path); assert cp["identity"] == ti
                entries, seconds = cp["entries"], cp["seconds"]
            limit = 128 if args.stop_at is None else args.stop_at
            if not len(entries) <= limit <= 128: raise ValueError("Nondecreasing readout budget required")
            start = time.monotonic()
            for index in range(len(entries), limit):
                entries.append(atom.fit_tree(frozen["model"].estimators_[index], z, event, w))
                if len(entries) % cfg["checkpoint_every_trees"] == 0 or len(entries) == limit:
                    temp = path.with_suffix(".tmp")
                    joblib.dump(dict(identity=ti, entries=entries, seconds=seconds+time.monotonic()-start), temp, compress=0)
                    os.replace(temp, path)
                    beat(state="fitting_readout", view=key, action=action, trees=len(entries))
            if len(entries) != 128:
                beat(state="pilot_readout_not_full_fit", view=key, action=action); return False
            prediction = atom.predict(frozen["model"], entries, z)["probability"]
            support = dict(rows=len(ids), known=int(pr["known"].sum()), effective_unique=int(use.sum()),
                zero_known=int(event.sum()), zero_effective=int((event & use).sum()),
                zero_effective_moving=int((event & use & moving).sum()),
                zero_effective_stopped=int((event & use & ~moving).sum()),
                effective_draws=int(w.sum()), zero_draws=int(w[event].sum()), unknown_draws=int(draws[~pr["known"]].sum()),
                training_weighted_brier=float(np.average((prediction[use]-event[use])**2, weights=w[use])))
            base.immutable_json(receipt, dict(identity=ti, trees=128, seconds=seconds+time.monotonic()-start,
                support=support, checkpoint=str(path.relative_to(ROOT)), checkpoint_sha256=base.file_digest(path),
                result_source="fresh_leaf_readout_fit_cached_verified_forest"))
    return True


def receipts(ctx, root):
    records = {}
    for key, action in ctx["refs"]:
        r = read(root/"readouts"/key/action/"complete.json")
        assert r["trees"] == 128 and r["identity"]["view"] == key and r["identity"]["action"] == action
        assert key.rsplit("_seed",1)[0] not in r["identity"]["training_sites"]
        assert r["identity"]["experiment_sha256"] == base.file_digest(root/"identity.json")
        assert r["identity"]["forest_sha256"] == ctx["refs"][key,action]["checkpoint_sha256"]
        assert base.file_digest(ROOT/r["checkpoint"]) == r["checkpoint_sha256"]
        records[key,action] = r
    assert len(records) == 36
    return records


def decide(ctx, root, args, beat):
    cfg, pc = ctx["cfg"], ctx["parent"]; ap = pc["pack"][1]; data = ap[1]
    fitted = receipts(ctx,root); archives = []; queries = 0
    for key in ap[4]:
        for action in cfg["actions"]:
            values,p,_,cutoff = base.causal_view(ap,key,action)
            ids = values["ids"]; g,s = data["geometry"][ids],data["scale"][ids]
            raw,d,_ = base.native_features(g,p,s)
            frozen = joblib.load(ROOT/ctx["refs"][key,action]["checkpoint"])
            cp = joblib.load(ROOT/fitted[key,action]["checkpoint"])
            assert cp["identity"] == fitted[key,action]["identity"]
            x = parent.features(raw,d,s,cutoff)
            estimate = atom.predict(frozen["model"],cp["entries"],standardized(x,frozen["preprocess"]))
            with np.load(ROOT/ctx["decisions"][key,action]["path"], allow_pickle=False) as z:
                np.testing.assert_array_equal(ids,z["ids"]); np.testing.assert_array_equal(d,z["distance"])
                f,original = z["fractions"].copy(),z["choices"].copy()
            assert (estimate["probability"] <= f[:,2]+2e-6).all()
            admit = atom.allowed(estimate["probability"],estimate["supported"])
            bank = base.scores(f,d,cutoff,g[:,:16].reshape(-1,8,2),cfg["easy_rho"])
            np.testing.assert_array_equal(original[:,0],bank["point"])
            bits = np.zeros((len(ids),6),bool); bits[:,0] = bank["point"] & admit
            ix = base.scene_index(data["recordings"][ids],data["frames"][ids],data["tracks"][ids])
            records = []
            for j in range(len(ix["offsets"])-1):
                q = ix["order"][ix["offsets"][j]:ix["offsets"][j+1]]
                gain,risk,den,ok = (bank[k][q] for k in ("gain","risk","denominator","eligible"))
                k = int(bits[q,0].sum()); pool = np.flatnonzero(bank["point"][q])
                order = pool[np.lexsort((ids[q[pool]],-gain[pool]))]; bits[q[order[:k]],3] = True
                reports = {}
                for col,rule in enumerate(("population","selected"),1):
                    rr = risk if col==1 else np.maximum(risk,0)-cfg["easy_rho"]*den
                    budget = cfg["easy_rho"]*float(den.sum()) if col==1 else 0.
                    guarded,rep = base.allocate(gain,rr,ok & admit[q],budget)
                    matched,mrep = atom.matched_with_incumbent(gain,rr,ok,budget,guarded,base.allocate)
                    bits[q,col],bits[q,col+3] = guarded,matched
                    reports[rule] = rep; reports["matched_"+rule] = mrep
                assert np.array_equal(bits[q,:3].sum(0),bits[q,3:].sum(0))
                records.append(dict(recording=str(ix["recordings"][j]),frame=int(ix["frames"][j]),reports=reports))
            path = root/"decisions"/f"{key}_{action}.npz"
            if args.verify and not path.exists(): raise ValueError("Replay cannot create missing choices")
            base.write_arrays(path,dict(ids=ids,choices=bits,original=original,probability=estimate["probability"],
                supported=estimate["supported"],minimum_unique_rows=estimate["minimum_unique_rows"]))
            rp = path.with_suffix(".json")
            base.immutable_json(rp,dict(experiment_sha256=base.file_digest(root/"identity.json"),view=key,action=action,
                path=str(path.relative_to(ROOT)),sha256=base.file_digest(path),queries=records,
                probability_positive_rows=int((estimate["probability"]>0).sum()), unsupported_rows=int((~estimate["supported"]).sum())))
            archives.append(dict(path=str(rp.relative_to(ROOT)),sha256=base.file_digest(rp)))
            queries += len(records); beat(state="decisions_saved",view=key,action=action,queries=queries)
    assert queries == 188388
    base.immutable_json(root/"decisions_complete.json",dict(experiment_sha256=base.file_digest(root/"identity.json"),
        archives=archives,queries=queries,outcomes_used_for_choices=False))
    if args.verify: base.immutable_json(ROOT/cfg["reports"]/"decision_replay.json",dict(exact_arrays=True,exact_reports=True))


def evaluate(ctx, root, args, beat):
    cfg,pc = ctx["cfg"],ctx["parent"]; ap = pc["pack"][1]; data = ap[1]; n = len(data["sites"])
    fitted = receipts(ctx,root); done = read(root/"decisions_complete.json"); refs={}; diagnostics=[]
    for ref in done["archives"]:
        assert base.file_digest(ROOT/ref["path"]) == ref["sha256"]
        r = read(ROOT/ref["path"]); assert base.file_digest(ROOT/r["path"]) == r["sha256"]
        refs[r["view"],r["action"]] = r
        diagnostics.extend(v for q in r["queries"] for v in q["reports"].values())
    policies = ["old_strict"]+[family+"_"+rule for family in ("cutoff","atom","matched") for rule in cfg["rules"]]
    summary,contrasts,quality = {},{},[]
    for action in cfg["actions"]:
        errors={p:[] for p in policies}; ends={p:[] for p in policies}; detail={p:{} for p in policies}
        for seed in cfg["seeds"]:
            ref=next(r for r in ap[3]["outcome_archives"] if r["path"].endswith(f"{action}_seed{seed}.npz"))
            assert base.file_digest(ROOT/ref["path"])==ref["sha256"]
            with np.load(ROOT/ref["path"],allow_pickle=False) as z: o={k:z[k].copy() for k in z.files}
            cv,cf=o["cv"],o["cf"]; masks={k:o[k] for k in ("complete","zero_CV","positive_easy","hard")}
            bits=np.zeros((n,10),bool); seen=np.zeros(n,bool)
            path=parent.PARENT/"outcomes"/f"{action}_seed{seed}.npz"
            assert base.file_digest(path)==ctx["identity"]["source_bindings"][str(path.relative_to(ROOT))]
            with np.load(path,allow_pickle=False) as z: bits[:,0]=z["choices"][:,pc["pcfg"]["policies"].index("old_strict")]
            for site in cfg["sites"]:
                key=f"{site}_seed{seed}"
                with np.load(ROOT/refs[key,action]["path"],allow_pickle=False) as z:
                    ids=z["ids"]; assert not seen[ids].any(); seen[ids]=True
                    bits[ids,1:4],bits[ids,4:]=z["original"],z["choices"]
                    k=masks["complete"][ids]; zero=masks["zero_CV"][ids]
                    moving=np.any(data["geometry"][ids,:16].reshape(-1,8,2)[:,-1] != data["geometry"][ids,:16].reshape(-1,8,2)[:,-2],axis=1)
                    quality.append(dict(view=key,action=action,known=int(k.sum()),zero=int(zero.sum()),
                        brier_complete=float(np.mean((z["probability"][k]-zero[k])**2)),
                        moving_zero=int((moving&zero).sum()),moving_zero_admitted=int((moving&zero&atom.allowed(z["probability"],z["supported"])).sum()),
                        probability_positive_rows=int((z["probability"]>0).sum())))
            assert seen.all()
            def metric(e,reference=cv,mask=None,ci=False):
                use=np.ones(n,bool) if mask is None else mask
                return base.paired_scene_metrics(e[use],reference[use],data["sites"][use],expected_scenes=cfg["sites"],
                    dataset="sdd",coordinate_unit="annotation_pixel",bootstrap_resamples=3000 if ci else 0)
            for j,p in enumerate(policies):
                selected=bits[:,j]; e=np.where(selected,o["candidate_ade"],cv); f=np.where(selected,o["candidate_fde"],cf)
                errors[p].append(e); ends[p].append(f)
                detail[p][str(seed)]=dict(ADE=metric(e),FDE=metric(f,cf),selected=int(selected.sum()),
                    subsets={k:metric(e,mask=m) for k,m in masks.items()},selected_unknown=int((selected&np.isnan(cv)).sum()),
                    selected_incomplete=int((selected&~masks["complete"]).sum()),zero_CV_harmed=int((selected&masks["zero_CV"]&(e>0)).sum()),
                    full_grid_gain_bounds={site:[float(np.where(selected,o[b],0)[data["sites"]==site].mean()) for b in ("lower","upper")] for site in cfg["sites"]})
            base.write_arrays(root/"outcomes"/f"{action}_seed{seed}.npz",dict(choices=bits))
        mean={p:np.mean(errors[p],axis=0) for p in policies}
        for p in policies:
            row=dict(ADE=metric(mean[p],ci=True),FDE=metric(np.mean(ends[p],axis=0),cf,ci=True),
                subsets={k:metric(mean[p],mask=m,ci=True) for k,m in masks.items()},seeds=detail[p])
            summary[action+"__"+p]=row
            if p=="old_strict" or p.startswith("cutoff_"): assert row==ctx["old"]["summary"][action+"__"+p]
        for rule in cfg["rules"]:
            for right in ("cutoff_"+rule,"matched_"+rule):
                contrast={}
                for subset,mask in (("all",None),("hard",masks["hard"]),("positive_easy",masks["positive_easy"])):
                    l,r=metric(mean["atom_"+rule],mask=mask)["by_scene"],metric(mean[right],mask=mask)["by_scene"]
                    contrast[subset]=base.paired_scene_contrast([l[s]["gain_percent"] for s in cfg["sites"]],
                        [r[s]["gain_percent"] for s in cfg["sites"]],resamples=3000)
                contrasts[action+"__atom_"+rule+"_minus_"+right]=contrast
        beat(state="aggregate_complete",action=action)
    public=ROOT/cfg["reports"]
    result=dict(result_sources=dict(leaf_readout_fits="fresh_run",forests_forecasts_targets="cached_verified",decisions_readout="fresh_run",external_evaluation="not_run"),
        experiment_sha256=base.file_digest(root/"identity.json"),decisions_sha256=base.file_digest(root/"decisions_complete.json"),
        rows=n,sites=cfg["sites"],seeds=cfg["seeds"],policies=policies,summary=summary,contrasts=contrasts,
        held_event_quality=quality,fits=[dict(view=k,action=a,**r) for (k,a),r in fitted.items()],old_summary_rows_exact=12,
        solver=dict(instances=len(diagnostics),optimal_unverified=sum(not r["optimal"] for r in diagnostics),
            incumbent_retained=sum(r["status"]=="feasible_incumbent_retained" for r in diagnostics),
            risk_violations=sum(not r["constraint_pass"] for r in diagnostics),count_mismatches=0),
        scope="design_exposed_source_only_not_confirmation",deployment=False,stage5c_executed=False,smc_enabled=False)
    if args.verify and not (public/"analysis.json").exists(): raise ValueError("Original aggregate required")
    base.immutable_json(public/"analysis.json",result)
    if args.verify: base.immutable_json(public/"aggregate_replay.json",dict(exact=True,analysis_sha256=base.file_digest(public/"analysis.json")))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--phase",choices=["preflight","train","decide","evaluate","all"],default="preflight")
    p.add_argument("--resume",action="store_true"); p.add_argument("--verify",action="store_true")
    p.add_argument("--view"); p.add_argument("--action"); p.add_argument("--stop-at",type=int)
    args=p.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg=read(ROOT/CONFIG); root=ROOT/cfg["output"]; root.mkdir(parents=True,exist_ok=True)
    def beat(**values):
        row=dict(pid=os.getpid(),updated_unix=time.time(),**values); base.json_write(root/"heartbeat.json",row)
        with (root/"events.jsonl").open("a") as f: f.write(json.dumps(row)+"\n")
        print(json.dumps(row),flush=True)
    with (root/"runner.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); beat(state="verifying_sources")
        ctx=load(); base.immutable_json(root/"identity.json",ctx["identity"])
        if args.view and args.view not in ctx["parent"]["pack"][1][4]: raise ValueError("Unknown view")
        if args.action and args.action not in cfg["actions"]: raise ValueError("Unknown action")
        beat(state="preflight_complete")
        if args.phase in ("train","all") and not train(ctx,root,args,beat): return
        if args.phase in ("decide","all"): decide(ctx,root,args,beat)
        if args.phase in ("evaluate","all"): evaluate(ctx,root,args,beat)
        base.assert_current(ctx["identity"]); beat(state="phase_complete",phase=args.phase)


if __name__ == "__main__": main()
