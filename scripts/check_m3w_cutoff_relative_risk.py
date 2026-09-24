"""Separate arithmetic on frozen cutoff-context fits and decisions."""
import json
import math
from pathlib import Path
import platform
import sys
if platform.system() == "Darwin" and platform.machine() != "arm64":
    raise RuntimeError("Native arm64 .venv-pytorch required")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
from src.evaluation.m3w_experiment_contract import file_digest
from scripts.run_m3w_native_forecast import immutable_json


def read(p): return json.loads((ROOT/p).read_text())


def main():
    cfg = read("configs/m3w_cutoff_relative_risk_v1.json")
    root,public = ROOT/cfg["output"],ROOT/cfg["reports"]
    a,identity,done = read(public/"analysis.json"),read(root/"identity.json"),read(root/"decisions_complete.json")
    assert a["experiment_sha256"] == file_digest(root/"identity.json")
    assert a["decisions_sha256"] == file_digest(root/"decisions_complete.json")
    parent = read("outputs/publication_readiness_2026_09/dimensionless_risk_v1/analysis.json")
    paired = 0
    for r in a["fits"]:
        assert file_digest(ROOT/r["checkpoint"]) == r["checkpoint_sha256"]
        cp = joblib.load(ROOT/r["checkpoint"])
        assert cp["identity"] == r["identity"] and len(cp["model"].estimators_) == 128
        assert cp["seed"] == int(r["view"].rsplit("_seed",1)[1])
        assert cp["draws"].sum() == 768000 and not cp["draws"][~cp["preprocess"]["known"]].any()
        assert {t.max_features_ for t in cp["model"].estimators_} == {118}
        assert r["view"].split("_seed")[0] not in cp["preprocess"]["training_sites"]
        old = next(x for x in parent["fits"] if x["view"]==r["view"] and x["action"]==r["action"] and x["arm"]=="native")
        for k in ("targets_sha256","draws_sha256","known_sha256","training_sites"):
            assert r["identity"][k] == old["identity"][k]
        oldtarget = read("data/stage_cvpr2027_experiments/net_easy_moment_v1/trials/"+r["view"]+"/"+r["action"]+"/complete.json")
        assert file_digest(ROOT/oldtarget["checkpoint"]) == old["identity"]["parent_fit_sha256"]
        target_cp = joblib.load(ROOT/oldtarget["checkpoint"])
        # The sampler's strata cutoff is not necessarily the target/budget cutoff.
        assert target_cp["identity"] == oldtarget["identity"]
        assert r["identity"]["cutoff"] == target_cp["identity"]["easy_cut"]
        paired += 1
    assert paired == 36
    manifest = read("data/stage_cvpr2027_experiments/sdd_auxiliary_v1/inputs/manifest.json")
    sites,recordings,frames = [],[],[]
    for r in manifest["records"]:
        site = r["recording"].split("/")[0]
        if site not in cfg["sites"]: continue
        path = "data/stage_cvpr2027_experiments/sdd_auxiliary_v1/inputs/"+r["recording"]+"/query_keys.npy"
        assert file_digest(ROOT/path) == identity["source_bindings"][path]
        keys = np.load(ROOT/path,allow_pickle=False)
        sites.extend([site]*len(keys)); recordings.extend([r["recording"]]*len(keys)); frames.extend(keys[:,0].tolist())
    sites,recordings,frames = np.array(sites),np.array(recordings),np.array(frames)
    assert len(sites) == 175756
    decisions = {}; constraints = 0; point_selected = 0
    for ref in done["archives"]:
        assert file_digest(ROOT/ref["path"]) == ref["sha256"]
        r = read(ref["path"]); assert file_digest(ROOT/r["path"]) == r["sha256"]
        decisions[r["view"],r["action"]] = r["path"]
        with np.load(ROOT/r["path"],allow_pickle=False) as z:
            ids,bits,f,d = z["ids"],z["choices"],z["fractions"],z["distance"]
            risk = (f[:,2]-f[:,3])*d; den = f[:,4]*r["cutoff"]
            basic = ((f[:,0]-f[:,1])*d>0)&(d>0)&(den>0)
            assert not (bits&~basic[:,None]).any()
            assert not (bits[:,0]&(risk>cfg["easy_rho"]*den)).any()
            point_selected += int(bits[:,0].sum())
            groups = {}
            for local,idx in enumerate(ids):
                groups.setdefault((str(recordings[idx]),int(frames[idx])),[]).append(local)
            assert len(groups) == len(r["queries"])
            for q in groups.values():
                q = np.asarray(q,dtype=np.int64)
                assert math.fsum(risk[q][bits[q,1]]) <= cfg["easy_rho"]*float(den[q].sum())
                assert math.fsum((np.maximum(risk[q],0)-cfg["easy_rho"]*den[q])[bits[q,2]]) <= 0
                constraints += 2
    outcomes = read("outputs/publication_readiness_2026_09/easy_moment_v1/analysis.json")
    reductions = 0
    for action in cfg["actions"]:
        errors = {p:[] for p in cfg["policies"]}
        for seed in cfg["seeds"]:
            ref = next(x for x in outcomes["outcome_archives"] if x["path"].endswith(f"{action}_seed{seed}.npz"))
            assert file_digest(ROOT/ref["path"]) == ref["sha256"]
            with np.load(ROOT/ref["path"],allow_pickle=False) as z: o = {k:z[k] for k in z.files}
            masks = {"all":np.ones(len(sites),bool),**{k:o[k] for k in ("complete","hard","positive_easy","zero_CV")}}
            bits = np.zeros((len(sites),3),bool)
            for site in cfg["sites"]:
                with np.load(ROOT/decisions[f"{site}_seed{seed}",action],allow_pickle=False) as z: bits[z["ids"]] = z["choices"]
            for j,p in enumerate(cfg["policies"]):
                e = np.where(bits[:,j],o["candidate_ade"],o["cv"]); errors[p].append(e)
                row = a["summary"][action+"__"+p]["seeds"][str(seed)]
                assert row["selected"] == int(bits[:,j].sum())
                assert row["zero_CV_harmed"] == int(((e>0)&o["zero_CV"]).sum())
        for p in cfg["policies"]:
            e = np.mean(errors[p],axis=0); record = a["summary"][action+"__"+p]
            for subset,mask in masks.items():
                metric = record["ADE"] if subset=="all" else record["subsets"][subset]
                for site in cfg["sites"]:
                    use = mask&(sites==site)&np.isfinite(o["cv"]); row = metric["by_scene"][site]
                    assert row["rows"] == int(use.sum())
                    if use.any():
                        np.testing.assert_allclose(e[use].mean(),row["model_error"],rtol=1e-12)
                        np.testing.assert_allclose(o["cv"][use].mean(),row["reference_error"],rtol=1e-12)
                    reductions += 1
    for key,row in parent["summary"].items(): assert row == a["summary"][key]
    immutable_json(public/"separate_checks.json",dict(result_source="fresh_run",method="separate_arithmetic_same_executor",
        code_sha256=file_digest(Path(__file__)),analysis_sha256=file_digest(public/"analysis.json"),
        matched_fit_checks=paired,query_constraint_checks=constraints,point_selected_checks=point_selected,
        scene_reductions=reductions,parent_rows_exact=30,predictive_independence=False,
        exhaustive_solver_optimality="not_run"))
    print(json.dumps(dict(fits=paired,constraints=constraints,reductions=reductions)))


if __name__ == "__main__": main()
