"""Matched source-only six-moment forests; decisions freeze before readout."""
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
from scripts import run_m3w_net_easy_moment_guarded as parent
from scripts.run_m3w_easy_allocation import causal_view
from scripts.run_m3w_bounded_cost import training_data, features as native_features, read_arrays
from scripts.run_m3w_native_forecast import array_hash, file_digest, immutable_json, json_write, assert_current
from scripts.run_m3w_native_nested import write_arrays
from src.evaluation.m3w_protected_motion_controls import causal_candidate, supported_costs
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
from src.evaluation.m3w_native_matched_coverage import paired_scene_contrast
from src.evaluation.m3w_native_scene_alignment import scene_index
from src.world_model.m3w_native_gain_harm import preprocess as native_preprocess
from src.world_model.m3w_dimensionless_risk import features, targets, preprocess, scores, ARMS, TARGETS
from src.world_model.m3w_net_easy_guard import allocate
from src.training.m3w_fraction_forest import fit, predict

CONFIG = "configs/m3w_dimensionless_risk_v1.json"
CODE = ("scripts/run_m3w_dimensionless_risk.py", "src/world_model/m3w_dimensionless_risk.py",
        "src/training/m3w_fraction_forest.py", "tests/test_m3w_dimensionless_risk.py")


def load():
    cfg = json.loads((ROOT/CONFIG).read_text())
    old_identity = ROOT/"data/stage_cvpr2027_experiments/net_easy_moment_guarded_v1/identity.json"
    old_analysis = ROOT/"outputs/publication_readiness_2026_09/net_easy_moment_guarded_v1/analysis.json"
    assert file_digest(old_identity) == cfg["parent_identity_sha256"]
    assert file_digest(old_analysis) == cfg["parent_analysis_sha256"]
    pack = parent.load()
    assert pack[-1] == json.loads(old_identity.read_text())
    assert cfg["feature_arms"] == list(ARMS) and cfg["forest"] == pack[0]["forest"]
    assert cfg["easy_rho"] == .02 and cfg["bootstrap_resamples"] == 3000
    for k in ("sites", "seeds", "actions"): assert cfg[k] == pack[0][k]
    assert not any(cfg[k] for k in ("new_predictor_training", "threshold_search", "model_selection",
        "external_readout", "independent_calibration", "confirmation", "deployment", "stage5c_executed", "smc_enabled"))
    refs = parent.receipts(pack)
    bindings = dict(pack[-1]["source_bindings"])
    for r in refs.values(): bindings[r["checkpoint"]] = r["checkpoint_sha256"]
    for path in (CONFIG, cfg["registration"], *CODE, str(old_identity.relative_to(ROOT)), str(old_analysis.relative_to(ROOT))):
        bindings[path] = file_digest(ROOT/path)
    identity = dict(config=cfg, source_bindings=bindings, targets=list(TARGETS),
        role="design_exposed_SDD_source_excluded_not_confirmation", numpy=np.__version__, torch=torch.__version__,
        sklearn=__import__("sklearn").__version__, runtime=dict(architecture=platform.machine(), torch_threads=4, interop_threads=1, num_workers=0))
    assert_current(identity)
    return cfg, pack, refs, identity


def prepare(pack, key, action, refs):
    _, data, ev, mv, ep, cuts, _, frozen, _, _ = pack[2]
    meta = ev[key] if action == "eqmotion" else mv[key]
    cp = parent.parent.old_head(frozen[key, action])
    ids, x, y, d, pr = training_data(meta, data)
    with np.load(ROOT/meta["targets_path"], allow_pickle=False) as z:
        np.testing.assert_array_equal(ids, z["ids"])
        cv = z["baseline_ade"].copy()
    cv[~pr["known"]] = np.nan
    if action == "damped_velocity_005":
        g, s = data["geometry"][ids], data["scale"][ids]
        p = causal_candidate(g, action)
        x, d, _ = native_features(g, p, s)
        valid, truth = read_arrays(data, ids, "valid"), read_arrays(data, ids, "target")
        cv, _ = native_errors(g[:, 332:356].reshape(-1, 12, 2), truth, valid, s)
        err, _ = native_errors(p, truth, valid, s)
        y, cv = supported_costs(err, cv, valid.all(1))
        pr = native_preprocess(x, y, cv, data["sites"][ids], meta["outer_site"])
    for k in ("mean", "std", "known", "weights", "constant"):
        np.testing.assert_array_equal(pr[k], cp["preprocess"][k])
    q = targets(y, cv, d, pr["known"], cuts[key]["positive_easy_cut"])
    old = refs[key, action]["identity"]
    assert array_hash(ids, x, d) == old["inputs_sha256"]
    assert array_hash(q[:, 2:]) == old["targets_sha256"]
    assert array_hash(cp["draws"]) == old["draws_sha256"]
    assert cp["draws"].sum() == 768000 and not cp["draws"][~pr["known"]].any()
    assert meta["outer_site"] not in pr["training_sites"]
    assert not (data["sites"][ids] == meta["outer_site"]).any()
    return ids, x, d, pr, q, cp["draws"], data["scale"][ids]


def train(cfg, pack, refs, root, args, beat):
    for key in pack[1][4]:
        if args.view and key != args.view: continue
        for action in cfg["actions"]:
            if args.action and action != args.action: continue
            ids, raw, d, pr, q, draws, s = prepare(pack, key, action, refs)
            for arm in ARMS:
                if args.arm and arm != args.arm: continue
                x = features(raw, d, s, arm)
                pp = preprocess(x, pr)
                ti = dict(experiment_sha256=file_digest(root/"identity.json"), view=key, action=action, arm=arm,
                    feature_width=x.shape[1], inputs_sha256=array_hash(ids, x, d), targets_sha256=array_hash(q),
                    draws_sha256=array_hash(draws), training_sites=pr["training_sites"],
                    known_sha256=array_hash(pr["known"]), parent_fit_sha256=refs[key, action]["checkpoint_sha256"])
                folder = root/"trials"/key/action/arm
                receipt = folder/"complete.json"
                if receipt.exists():
                    r = json.loads(receipt.read_text())
                    assert r["identity"] == ti and r["fit"]["complete"]
                    assert file_digest(ROOT/r["checkpoint"]) == r["checkpoint_sha256"]
                    beat(state="cached_verified_fit", view=key, action=action, arm=arm)
                    continue
                result = fit(x, q, d, pp, draws, settings=cfg["forest"], seed=int(key.rsplit("_seed", 1)[1]),
                    identity=ti, directory=folder, heartbeat=lambda **v: beat(view=key, action=action, arm=arm, **v),
                    resume=args.resume, stop_at=args.stop_at)
                if not result["complete"]:
                    beat(state="pilot_checkpoint_not_full_matrix", view=key, action=action, arm=arm)
                    return False
                cp = folder/"checkpoint.joblib"
                immutable_json(receipt, dict(identity=ti, fit=result, checkpoint=str(cp.relative_to(ROOT)),
                    checkpoint_sha256=file_digest(cp), result_source="fresh_run"))
    return True


def receipts(cfg, root, keys):
    out = {}
    for key in keys:
        for action in cfg["actions"]:
            for arm in ARMS:
                r = json.loads((root/"trials"/key/action/arm/"complete.json").read_text())
                ti = r["identity"]
                assert ti["experiment_sha256"] == file_digest(root/"identity.json")
                assert ti["view"] == key and ti["action"] == action and ti["arm"] == arm
                assert key.rsplit("_seed", 1)[0] not in ti["training_sites"]
                assert r["fit"]["complete"] and r["fit"]["trees"] == 128
                assert file_digest(ROOT/r["checkpoint"]) == r["checkpoint_sha256"]
                out[key, action, arm] = r
    assert len(out) == 72
    return out


def decide(cfg, pack, root, args, beat):
    ap = pack[1]; data = ap[1]
    fitted = receipts(cfg, root, ap[4]); archives = []; count = 0
    for key in ap[4]:
        for action in cfg["actions"]:
            values, p, _, cutoff = causal_view(ap, key, action)
            ids = values["ids"]
            g, s = data["geometry"][ids], data["scale"][ids]
            raw, d, _ = native_features(g, p, s)
            bits = {name: np.zeros(len(ids), bool) for name in cfg["policies"]}
            bits["uncontrolled"][:] = True
            bits["old_strict"] = values[action+"__strict_stop"].copy()
            bank, fields = {}, dict(ids=ids)
            for arm in ARMS:
                r = fitted[key, action, arm]
                cp = joblib.load(ROOT/r["checkpoint"])
                assert cp["identity"] == r["identity"] and cp["settings"] == cfg["forest"]
                assert cp["seed"] == int(key.rsplit("_seed", 1)[1])
                assert not cp["draws"][~cp["preprocess"]["known"]].any()
                x = features(raw, d, s, arm)
                f = predict(cp["model"], x, cp["preprocess"])
                bank[arm] = scores(f, d, cutoff, g[:, :16].reshape(-1, 8, 2), cfg["easy_rho"])
                fields[arm+"_fractions"] = f
                bits[arm+"_point"] = bank[arm]["point"]
                if arm == "dimensionless":
                    # Metadata-unit relabel only: normalized forecasts stay fixed.
                    for factor in (.01, 100.):
                        altered = raw.copy()
                        altered[:, 354] = np.log(s*factor)
                        altered[:, 355] = np.log1p(d*factor)
                        xx = features(altered, d*factor, s*factor, arm)
                        np.testing.assert_allclose(xx, x, rtol=1e-7, atol=1e-7)
                        ff = predict(cp["model"], xx, cp["preprocess"])
                        np.testing.assert_array_equal(ff, f)
            index = scene_index(data["recordings"][ids], data["frames"][ids], data["tracks"][ids])
            reports = []
            for j in range(len(index["offsets"])-1):
                rows = index["order"][index["offsets"][j]:index["offsets"][j+1]]
                report = dict(recording=str(index["recordings"][j]), frame=int(index["frames"][j]))
                for arm in ARMS:
                    b = bank[arm]; gain, risk, den, ok = (b[k][rows] for k in ("gain", "risk", "denominator", "eligible"))
                    v, r = allocate(gain, risk, ok, cfg["easy_rho"]*float(den.sum()), seconds=cfg["solver_seconds"])
                    bits[arm+"_population"][rows] = v; report[arm+"_population"] = r
                    v, r = allocate(gain, np.maximum(risk, 0)-cfg["easy_rho"]*den, ok, 0., seconds=cfg["solver_seconds"])
                    bits[arm+"_selected"][rows] = v; report[arm+"_selected"] = r
                b = bank["dimensionless"]; native_count = int(bits["native_population"][rows].sum())
                if native_count > int(b["eligible"][rows].sum()):
                    r = dict(status="insufficient_support_floor", optimal=False, exact_count_pass=False, constraint_pass=True)
                    v = np.zeros(len(rows), bool)
                else:
                    v, r = allocate(b["gain"][rows], b["risk"][rows], b["eligible"][rows],
                        cfg["easy_rho"]*float(b["denominator"][rows].sum()), count=native_count, seconds=cfg["solver_seconds"])
                bits["dimensionless_matched_native_population"][rows] = v
                report["matched"] = r; reports.append(report)
            fields.update(choices=np.column_stack([bits[k] for k in cfg["policies"]]), distance=d,
                          native_gain=bank["native"]["gain"], dimensionless_gain=bank["dimensionless"]["gain"])
            path = root/"decisions"/f"{key}_{action}.npz"
            if args.verify and not path.exists(): raise ValueError("Replay cannot create missing decisions")
            write_arrays(path, fields)
            rp = path.with_suffix(".json")
            immutable_json(rp, dict(experiment_sha256=file_digest(root/"identity.json"), view=key, action=action,
                path=str(path.relative_to(ROOT)), sha256=file_digest(path), queries=reports))
            archives.append(dict(path=str(rp.relative_to(ROOT)), sha256=file_digest(rp)))
            count += len(reports)
            beat(state="decisions_saved", view=key, action=action, query_action_seed_instances=count)
    assert count == 188388
    immutable_json(root/"decisions_complete.json", dict(experiment_sha256=file_digest(root/"identity.json"),
        archives=archives, queries=count, fits=72, outcomes_used_for_choices=False))
    if args.verify: immutable_json(ROOT/cfg["reports"]/"decision_replay.json", dict(exact_arrays=True, exact_reports=True))


def evaluate(cfg, pack, root, args, beat):
    ap = pack[1]; data = ap[1]; n = len(data["sites"])
    fitted = receipts(cfg, root, ap[4])
    complete = json.loads((root/"decisions_complete.json").read_text())
    assert complete["experiment_sha256"] == file_digest(root/"identity.json")
    decisions = {}; diagnostics = []
    for ref in complete["archives"]:
        assert file_digest(ROOT/ref["path"]) == ref["sha256"]
        r = json.loads((ROOT/ref["path"]).read_text())
        assert file_digest(ROOT/r["path"]) == r["sha256"]
        decisions[r["view"], r["action"]] = r
        diagnostics.extend(q[k] for q in r["queries"] for k in ("native_population", "native_selected", "dimensionless_population", "dimensionless_selected", "matched"))
    summary, contrasts, quality = {}, {}, []
    for action in cfg["actions"]:
        errors = {p: [] for p in cfg["policies"]}; ends = {p: [] for p in cfg["policies"]}
        detail = {p: {} for p in cfg["policies"]}
        for seed in cfg["seeds"]:
            ref = next(r for r in ap[3]["outcome_archives"] if r["path"].endswith(f"{action}_seed{seed}.npz"))
            assert file_digest(ROOT/ref["path"]) == ref["sha256"]
            with np.load(ROOT/ref["path"], allow_pickle=False) as z: o = {k:z[k].copy() for k in z.files}
            cv, cf = o["cv"], o["cf"]; masks = {k:o[k] for k in ("complete", "zero_CV", "positive_easy", "hard")}
            choices = np.zeros((n, len(cfg["policies"])), bool); seen = np.zeros(n, bool)
            for key in ap[4]:
                if not key.endswith(f"seed{seed}"): continue
                r = decisions[key, action]
                with np.load(ROOT/r["path"], allow_pickle=False) as z:
                    ids = z["ids"].copy(); choices[ids] = z["choices"]; assert not seen[ids].any(); seen[ids] = True
                    err = o["candidate_ade"][ids]
                    costs, reference = supported_costs(err, cv[ids], masks["complete"][ids])
                    cutoff = pack[2][5][key]["positive_easy_cut"]
                    q = targets(costs, reference, z["distance"], masks["complete"][ids], cutoff)
                    for arm in ARMS:
                        supported = masks["complete"][ids]
                        quality.append(dict(view=key, action=action, arm=arm, complete_rows=int(supported.sum()),
                            held_source_fraction_mse=((z[arm+"_fractions"][supported]-q[supported])**2).mean(0).tolist()))
            assert seen.all()
            def metric(e, reference=cv, mask=None, ci=False):
                use = np.ones(n, bool) if mask is None else mask
                return paired_scene_metrics(e[use], reference[use], data["sites"][use], expected_scenes=cfg["sites"],
                    dataset="sdd", coordinate_unit="annotation_pixel", bootstrap_resamples=3000 if ci else 0)
            for j, p in enumerate(cfg["policies"]):
                bits = choices[:, j]; e = np.where(bits, o["candidate_ade"], cv); f = np.where(bits, o["candidate_fde"], cf)
                errors[p].append(e); ends[p].append(f)
                detail[p][str(seed)] = dict(ADE=metric(e), FDE=metric(f, cf), selected=int(bits.sum()),
                    subsets={k:metric(e, mask=m) for k,m in masks.items()},
                    selected_unknown=int((bits & np.isnan(cv)).sum()), selected_incomplete=int((bits & ~masks["complete"]).sum()),
                    zero_CV_harmed=int((bits & masks["zero_CV"] & (e > 0)).sum()),
                    full_grid_gain_bounds={site:[float(np.where(bits,o[b],0)[data["sites"]==site].mean()) for b in ("lower","upper")] for site in cfg["sites"]})
            write_arrays(root/"outcomes"/f"{action}_seed{seed}.npz", dict(choices=choices))
        mean = {p:np.mean(errors[p], axis=0) for p in cfg["policies"]}
        for p in cfg["policies"]:
            summary[action+"__"+p] = dict(ADE=metric(mean[p], ci=True), FDE=metric(np.mean(ends[p],axis=0),cf,ci=True),
                subsets={k:metric(mean[p],mask=m,ci=True) for k,m in masks.items()}, seeds=detail[p])
        pairs = [("dimensionless_population","native_population"), ("dimensionless_point","native_point"),
                 ("dimensionless_selected","native_selected"), ("dimensionless_matched_native_population","native_population")]
        pairs += [(f"{arm}_{policy}", "old_strict") for arm in ARMS for policy in ("point","population","selected")]
        for left,right in pairs:
            contrast = {}
            for subset,mask in (("all",None),("hard",masks["hard"]),("positive_easy",masks["positive_easy"])):
                l,r = metric(mean[left],mask=mask)["by_scene"],metric(mean[right],mask=mask)["by_scene"]
                contrast[subset] = paired_scene_contrast([l[s]["gain_percent"] for s in cfg["sites"]],
                    [r[s]["gain_percent"] for s in cfg["sites"]],resamples=3000)
            contrasts[action+"__"+left+"_minus_"+right] = contrast
        beat(state="aggregate_complete", action=action)
    result = dict(result_source="fresh_head_training_and_decisions_cached_verified_predictors_and_targets",
        experiment_sha256=file_digest(root/"identity.json"), decisions_sha256=file_digest(root/"decisions_complete.json"),
        rows=n, sites=cfg["sites"], seeds=cfg["seeds"], policies=cfg["policies"], targets=list(TARGETS),
        summary=summary, contrasts=contrasts, held_source_fit_quality=quality,
        solver=dict(instances=len(diagnostics), optimal_unverified=sum(not r["optimal"] for r in diagnostics),
            exact_count_failures=sum(not r.get("exact_count_pass",True) for r in diagnostics),
            risk_violations=sum(not r["constraint_pass"] for r in diagnostics)),
        fits=[dict(view=k,action=a,arm=b,**r) for (k,a,b),r in fitted.items()],
        scope="design_exposed_source_only_not_confirmation", deployment=False,
        external_readout=False, stage5c_executed=False, smc_enabled=False)
    public = ROOT/cfg["reports"]
    if args.verify and not (public/"analysis.json").exists(): raise ValueError("Missing original aggregate")
    immutable_json(public/"analysis.json", result)
    if args.verify: immutable_json(public/"aggregate_replay.json",dict(exact=True,analysis_sha256=file_digest(public/"analysis.json")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["preflight","train","decide","evaluate","all"], default="preflight")
    p.add_argument("--resume", action="store_true"); p.add_argument("--verify", action="store_true")
    p.add_argument("--view"); p.add_argument("--action"); p.add_argument("--arm",choices=list(ARMS)); p.add_argument("--stop-at",type=int)
    args = p.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg = json.loads((ROOT/CONFIG).read_text()); root = ROOT/cfg["output"]; root.mkdir(parents=True,exist_ok=True)
    def beat(**v):
        e = dict(pid=os.getpid(),updated_unix=time.time(),**v); json_write(root/"heartbeat.json",e)
        with (root/"events.jsonl").open("a") as f: f.write(json.dumps(e)+"\n")
        print(json.dumps(e),flush=True)
    with (root/"runner.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        beat(state="verifying_source_dependencies")
        cfg,pack,refs,identity = load(); immutable_json(root/"identity.json",identity)
        if args.view and args.view not in pack[1][4]: raise ValueError("Unknown view")
        if args.action and args.action not in cfg["actions"]: raise ValueError("Unknown action")
        beat(state="preflight_complete",rows=len(pack[1][1]["sites"]))
        if args.phase in ("train","all"):
            if not train(cfg,pack,refs,root,args,beat): return
        if args.phase in ("decide","all"): decide(cfg,pack,root,args,beat)
        if args.phase in ("evaluate","all"): evaluate(cfg,pack,root,args,beat)
        assert_current(identity)
        beat(state="phase_complete",phase=args.phase)


if __name__ == "__main__": main()
