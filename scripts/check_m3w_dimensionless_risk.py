"""Separate post-freeze arithmetic: identities, decisions, bounds and reductions."""
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
if platform.system() == "Darwin" and platform.machine() != "arm64":
    raise RuntimeError("Native arm64 environment required")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from src.evaluation.m3w_experiment_contract import file_digest
from scripts.run_m3w_native_forecast import immutable_json


def read(p): return json.loads((ROOT/p).read_text())


def main():
    cfg = read("configs/m3w_dimensionless_risk_v1.json")
    root, public = ROOT/cfg["output"], ROOT/cfg["reports"]
    identity, analysis = read(root/"identity.json"), read(public/"analysis.json")
    assert analysis["experiment_sha256"] == file_digest(root/"identity.json")
    completed = read(root/"decisions_complete.json")
    assert analysis["decisions_sha256"] == file_digest(root/"decisions_complete.json")
    manifest = read("data/stage_cvpr2027_experiments/sdd_auxiliary_v1/inputs/manifest.json")
    sites, frames, recordings = [], [], []
    for r in manifest["records"]:
        site = r["recording"].split("/")[0]
        if site not in cfg["sites"]: continue
        path = "data/stage_cvpr2027_experiments/sdd_auxiliary_v1/inputs/"+r["recording"]+"/query_keys.npy"
        assert file_digest(ROOT/path) == identity["source_bindings"][path]
        keys = np.load(ROOT/path, allow_pickle=False)
        sites.extend([site]*len(keys)); recordings.extend([r["recording"]]*len(keys)); frames.extend(keys[:, 0].tolist())
    sites, frames, recordings = np.array(sites), np.array(frames), np.array(recordings)
    assert len(sites) == 175756
    paired = {}; fit_checks = 0
    for r in analysis["fits"]:
        assert file_digest(ROOT/r["checkpoint"]) == r["checkpoint_sha256"]
        cp = joblib.load(ROOT/r["checkpoint"])
        assert cp["identity"] == r["identity"] and len(cp["model"].estimators_) == 128
        assert cp["draws"].sum() == 768000 and not cp["draws"][~cp["preprocess"]["known"]].any()
        assert r["view"].split("_seed")[0] not in cp["preprocess"]["training_sites"]
        assert {t.max_features_ for t in cp["model"].estimators_} == {118}
        k = r["view"], r["action"]
        sig = {n:r["identity"][n] for n in ("targets_sha256", "draws_sha256", "known_sha256", "training_sites")}
        if k in paired: assert paired[k] == sig
        paired[k] = sig
        fit_checks += 1
    assert fit_checks == 72 and len(paired) == 36
    counts, violations, count_failures = 0, 0, 0
    decision_paths = {}
    for rr in completed["archives"]:
        assert file_digest(ROOT/rr["path"]) == rr["sha256"]
        r = read(rr["path"]); assert file_digest(ROOT/r["path"]) == r["sha256"]
        decision_paths[r["view"],r["action"]] = r["path"]
        with np.load(ROOT/r["path"],allow_pickle=False) as z:
            ids, bits, d = z["ids"], z["choices"], z["distance"]
            grouped = {}
            for local, idx in enumerate(ids):
                grouped.setdefault((str(recordings[idx]), int(frames[idx])), []).append(local)
            cutoff = None
            # Read only the frozen training cutoff, never infer it from held errors.
            old = read("data/stage_cvpr2027_experiments/net_easy_moment_v1/trials/"+r["view"]+"/"+r["action"]+"/complete.json")
            cutoff = old["identity"]["easy_cut"]
            for query in r["queries"]:
                q = np.asarray(grouped[query["recording"], query["frame"]], dtype=np.int64)
                assert len(q)
                for arm in ("native","dimensionless"):
                    f = z[arm+"_fractions"][q]
                    risk = (f[:,2]-f[:,3])*d[q]; den = f[:,4]*cutoff
                    for policy in ("population","selected"):
                        pick = bits[q,cfg["policies"].index(arm+"_"+policy)]
                        lhs = math.fsum(risk[pick]) if policy=="population" else math.fsum((np.maximum(risk,0)-cfg["easy_rho"]*den)[pick])
                        rhs = cfg["easy_rho"]*float(den.sum()) if policy=="population" else 0.
                        violations += int(lhs > rhs)
                        counts += 1
                matched = bits[q,cfg["policies"].index("dimensionless_matched_native_population")]
                native = bits[q,cfg["policies"].index("native_population")]
                fail = matched.sum() != native.sum()
                assert bool(fail) == (not query["matched"]["exact_count_pass"])
                count_failures += int(fail)
    assert not violations
    parent = read("outputs/publication_readiness_2026_09/easy_moment_v1/analysis.json")
    reductions = 0
    for action in cfg["actions"]:
        mean_errors = {p:[] for p in cfg["policies"]}
        masks = None
        for seed in cfg["seeds"]:
            ref = next(r for r in parent["outcome_archives"] if r["path"].endswith(f"{action}_seed{seed}.npz"))
            assert file_digest(ROOT/ref["path"] ) == ref["sha256"]
            with np.load(ROOT/ref["path"],allow_pickle=False) as z: o = {k:z[k] for k in z.files}
            masks = {"all":np.ones(len(sites),bool), **{k:o[k] for k in ("complete","hard","positive_easy","zero_CV")}}
            choice = np.zeros((len(sites),len(cfg["policies"])),bool)
            for site in cfg["sites"]:
                with np.load(ROOT/decision_paths[f"{site}_seed{seed}",action],allow_pickle=False) as z: choice[z["ids"]] = z["choices"]
            for j,p in enumerate(cfg["policies"]):
                e = np.where(choice[:,j],o["candidate_ade"],o["cv"]); mean_errors[p].append(e)
                official = analysis["summary"][action+"__"+p]["seeds"][str(seed)]
                assert official["selected"] == int(choice[:,j].sum())
                assert official["zero_CV_harmed"] == int(((e>0)&o["zero_CV"]).sum())
        for p in cfg["policies"]:
            m = np.mean(mean_errors[p],axis=0)
            record = analysis["summary"][action+"__"+p]
            for subset,mask in masks.items():
                metric = record["ADE"] if subset=="all" else record["subsets"][subset]
                for site in cfg["sites"]:
                    use = mask & (sites==site) & np.isfinite(o["cv"])
                    row = metric["by_scene"][site]
                    assert row["rows"] == int(use.sum())
                    if use.any():
                        mm,rr = float(m[use].mean()),float(o["cv"][use].mean())
                        np.testing.assert_allclose(mm,row["model_error"],rtol=1e-12)
                        np.testing.assert_allclose(rr,row["reference_error"],rtol=1e-12)
                        if rr>0: np.testing.assert_allclose(100*(1-mm/rr),row["gain_percent"],rtol=1e-12,atol=1e-12)
                    reductions += 1
    immutable_json(public/"separate_checks.json",dict(method="separate_arithmetic_same_agent_not_independent_confirmation",
        result_source="fresh_run", code_sha256=file_digest(Path(__file__)), analysis_sha256=file_digest(public/"analysis.json"),
        fit_checks=fit_checks, paired_target_draw_checks=len(paired), constraint_checks=counts,
        constraint_violations=violations, observed_exact_count_failures=count_failures, scene_reductions=reductions,
        exhaustive_solver_optimality="not_run", predictive_independence=False))
    print(json.dumps(dict(fit_checks=fit_checks,constraint_checks=counts,reductions=reductions,count_failures=count_failures)))


if __name__ == "__main__": main()
