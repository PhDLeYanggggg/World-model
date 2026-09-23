"""Freeze the complete inference chain and replay source-only, label-free probes."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == "Darwin" and platform.machine() != "arm64":
    raise RuntimeError("Native arm64 Python required")
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
import sklearn
import torch

from scripts.run_m3w_native_forecast import array_hash, assert_current, immutable_json, json_write
from scripts.build_m3w_native_scene_context import load_past_queries
from src.evaluation.m3w_recording_lineage import sha256
from src.data_unification.m3w_external_prefix_adapter import ExternalPrefixAdapter, dense_prefix
from src.world_model.m3w_frozen_policy_chain import SceneInput, FrozenView, scene_from_prefix, restore

CONFIG = "configs/m3w_external_policy_chain_v1.json"
CONTEXT = "outputs/publication_readiness_2026_09/native_scene_context_v2/analysis.json"
ALIGNMENT = "outputs/publication_readiness_2026_09/native_scene_alignment_v1/analysis.json"
CODE = ("scripts/freeze_m3w_external_policy_chain.py", "src/world_model/m3w_frozen_policy_chain.py",
    "tests/test_m3w_frozen_policy_chain.py", "src/data_unification/m3w_external_prefix_adapter.py",
    "tests/test_m3w_external_prefix_adapter.py", "src/world_model/m3w_native_joint_controls.py",
    "src/world_model/m3w_joint_intervention.py", "src/world_model/m3w_interaction_controls.py",
    "tests/test_m3w_native_joint_controls.py", "tests/test_m3w_interaction_controls.py",
    "scripts/build_m3w_native_scene_context.py")
CONTEXT_FIELDS = ("context_frame_ids", "context_agent_ids", "context_target_rows", "context_xy",
                  "context_cv_rollout", "context_cv_valid")


def specification():
    cfg = json.loads((ROOT/CONFIG).read_text())
    expected_policy = dict(harm_to_benefit_max=.1, exact_past_stop_guard=True,
        positive_net_gain_required=True, budgets=dict(full=1., half=.5),
        reference="top_predicted_net_gain_then_agent_id", harm_budget="mean_predicted_harm_of_reference",
        geometry=dict(past_radius="median_target_past_scale", threshold_fraction=.1, pair_weight=1.),
        solver_seconds=5., solver_failure="floor_and_mark_unmatched_not_optimal",
        context="all_visible_agents_with_explicit_CV_support_unknown_edges_unpriced_reported")
    if (cfg["policy"] != expected_policy or cfg["families"] != ["transformer", "eqmotion"]
            or cfg["heads"] != ["bounded_fraction", "matched_fraction_forest"] or cfg["seeds"] != [17, 29, 43]
            or cfg["source_probes"]["queries_per_recording"] != 3 or cfg["source_probes"]["future_labels_read"]
            or cfg["observed_steps"] != 8 or cfg["predicted_steps"] != 12
            or cfg["source_raw_annotation_stride"] != 12 or cfg["external_raw_annotation_stride"] != 1
            or any(cfg[k] for k in ("equal_physical_duration_claim", "new_training", "threshold_search",
                "model_selection", "reserved_source_readout", "risk_calibration", "independent_confirmation",
                "deployment", "stage5c_executed", "smc_enabled"))):
        raise ValueError("Fixed label-free protocol required")
    bindings = {}
    def bind(path, expected=None):
        path = str(path)
        value = sha256(ROOT/path)
        if (expected is not None and value != expected) or (path in bindings and bindings[path] != value):
            raise ValueError("Changed chain dependency: "+path)
        bindings[path] = value
    reports = {}
    for name in ("predictors", "costs"):
        ref = cfg[name]; bind(ref["path"], ref["sha256"])
        report = json.loads((ROOT/ref["path"]).read_text())
        for path, h in report["identity"]["source_bindings"].items():
            bind(path, h)
        replay_path = str(Path(ref["path"]).with_name("replay.json")); bind(replay_path)
        proof = json.loads((ROOT/replay_path).read_text())
        if not proof["all_checks_passed"] or proof["analysis_sha256"] != ref["sha256"]:
            raise ValueError("Unverified fitted bank")
        for r in report["models"]:
            bind(r["checkpoint"], r["checkpoint_sha256"])
        reports[name] = report
    independent = str(Path(cfg["costs"]["path"]).with_name("independent_verification.json"))
    bind(independent)
    proof = json.loads((ROOT/independent).read_text())
    if not proof["all_checks_passed"] or proof["analysis_sha256"] != cfg["costs"]["sha256"]:
        raise ValueError("Cost-bank independent arithmetic check missing")
    bind(CONTEXT, "ea58e42239180cd326cac5c22294b3fbf1668c484764d212d504bb400a99c17e")
    bind(ALIGNMENT, "8a5879cfc511943bb303e653e7a5aac0b8b3434792eaaee74c48e1bb5fbdee7f")
    context = json.loads((ROOT/CONTEXT).read_text())
    context_proof = str(Path(CONTEXT).with_name("verification.json")); bind(context_proof)
    cp = json.loads((ROOT/context_proof).read_text())
    if not cp["all_checks_passed"] or cp["analysis_sha256"] != bindings[CONTEXT]:
        raise ValueError("Context identity verification missing")
    for p, h in context["source_bindings"].items():
        bind(p, h)
    for r in context["records"]:
        bind(r["cache"]["path"], r["cache"]["sha256"])
    alignment = json.loads((ROOT/ALIGNMENT).read_text())["cache"]
    bind(alignment["path"], alignment["sha256"])
    for p in (CONFIG, cfg["registration"], *CODE):
        bind(p)
    if len(reports["predictors"]["models"]) != 6 or len(reports["costs"]["models"]) != 12:
        raise ValueError("All registered predictors and cost heads required")
    manifest = dict(config=cfg, source_bindings=bindings,
        predictors=[{k: r[k] for k in ("family", "seed", "checkpoint", "checkpoint_sha256")}
                    for r in reports["predictors"]["models"]],
        cost_heads=[{k: r[k] for k in ("view", "head", "checkpoint", "checkpoint_sha256")}
                    for r in reports["costs"]["models"]],
        architectures=reports["predictors"]["identity"]["architectures"],
        torch=torch.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__, numpy=np.__version__,
        architecture=platform.machine(), source_admission=False,
        risk_calibration="not_run", independent_confirmation="not_run", deployment=False)
    return cfg, manifest, context, alignment


def source_inputs(manifest, context, alignment):
    bindings = {}
    data = load_past_queries(bindings)
    for path, digest in bindings.items():
        if manifest["source_bindings"].get(path) != digest:
            raise ValueError("Input geometry not bound into the frozen chain")
    if array_hash(data["recordings"], data["tracks"], data["frames"]) != alignment["population_sha256"]:
        raise ValueError("Wrong source input row alignment")
    with np.load(ROOT/alignment["path"], allow_pickle=False) as z:
        transforms = {k: z[k] for k in ("origin", "rotation", "stored_metric_scale")}
    scenes = []
    for record in context["records"]:
        with np.load(ROOT/record["cache"]["path"], allow_pickle=False) as z:
            c = {k: z[k] for k in CONTEXT_FIELDS}
        frames = np.unique(c["context_frame_ids"])
        for frame in frames[[0, len(frames)//2, -1]]:
            rows = np.flatnonzero(c["context_frame_ids"] == frame)
            target = c["context_target_rows"][rows] >= 0
            tid = c["context_target_rows"][rows][target]
            ids = c["context_agent_ids"][rows]
            if (np.any(data["recordings"][tid] != record["recording"]) or np.any(data["frames"][tid] != frame)
                    or not np.array_equal([int(v.rsplit(":", 1)[1]) for v in data["tracks"][tid]], ids[target])):
                raise ValueError("Source agent/frame identity mismatch")
            g = data["geometry"][tid]
            origin, rotation, scale = [transforms[k][tid] for k in ("origin", "rotation", "stored_metric_scale")]
            base = c["context_cv_rollout"][rows].copy()
            base[target] = restore(g[:, 332:356].reshape(-1, 12, 2), origin, rotation, scale)
            s = SceneInput(record["recording"], int(frame), 12, ids.copy(), ids[target].copy(),
                g.copy(), origin.copy(), rotation.copy(), scale.copy(), c["context_xy"][rows].copy(),
                base, c["context_cv_valid"][rows].all(1))
            s.validate(); scenes.append(s)
    if len(scenes) != 99 or len(context["records"]) != 33:
        raise ValueError("Incomplete fixed source query coverage")
    return scenes


def input_hash(scene):
    return array_hash(scene.agent_ids, scene.target_ids, scene.geometry, scene.origins, scene.rotations,
                      scene.scales, scene.current, scene.baseline, scene.cv_valid)


def summarize(scene, out):
    decisions = np.column_stack([out["choices"][k] for k in sorted(out["choices"])])
    return dict(recording=scene.recording_id, frame=scene.frame_id, input_sha256=input_hash(scene),
        visible_agents=len(scene.agent_ids), neural_targets=len(scene.target_ids),
        unknown_CV_context=int((~scene.cv_valid).sum()), eligible=int(out["eligible"].sum()),
        rejected_model_outputs=int((~out["model_output_supported"]).sum()),
        forecast_score_sha256=array_hash(out["baseline"], out["candidate"], out["costs"]),
        decision_sha256=array_hash(decisions), switch_counts={k:int(v.sum()) for k,v in out["choices"].items()},
        budgets=out["budgets"], geometry=out["geometry"], future_labels_read=False)


def synthetic_inputs():
    t = np.arange(20)
    xy = np.stack([np.column_stack((.3*t+a, .05*t+a/3)) for a in range(4)])
    mask = np.ones((4, 20), bool); mask[-1, :7] = False
    p = dense_prefix(xy, mask, np.arange(4), query_frame=7)
    xy[:, 8:] += 1e6; mask[:, 8:] = False
    other = dense_prefix(xy, mask, np.arange(4), query_frame=7)
    a, b = [scene_from_prefix(ExternalPrefixAdapter(v, query_frame=7, recording_id="synthetic_prefix")) for v in (p, other)]
    if input_hash(a) != input_hash(b) or a.cv_valid[-1]:
        raise ValueError("Future tail changed input or unsupported context became certain")
    return a, b


def run(cfg, manifest, scenes, root, public, beat, verify):
    digest = sha256(public/"policy_manifest.json")
    results = []
    for p in manifest["predictors"]:
        view = f"{p['family']}_seed{p['seed']}"
        for h in [r for r in manifest["cost_heads"] if r["view"] == view]:
            key = view+"_"+h["head"]
            path = root/"probes"/(key+".json")
            if verify and not path.exists():
                raise ValueError("Verification cannot create missing chain probes")
            if path.exists() and not verify:
                r = json.loads(path.read_text())
                if r["manifest_sha256"] != digest:
                    raise ValueError("Changed cached probe identity")
                results.append(r); beat(state="cached_probe", view=key); continue
            model = FrozenView(ROOT, p, h, manifest["architectures"][p["family"]], p["seed"])
            queries = []
            started = time.monotonic()
            for i, scene in enumerate(scenes):
                queries.append(summarize(scene, model.infer(scene)))
                if i % 33 == 0:
                    beat(state="chain_probe", view=key, queries=i+1)
            a, b = synthetic_inputs()
            sa, sb = summarize(a, model.infer(a)), summarize(b, model.infer(b))
            if sa != sb:
                raise ValueError("Future perturbation changed the frozen chain")
            r = dict(view=key, manifest_sha256=digest, queries=queries, synthetic_prefix=sa,
                external_data_opened=False, future_target_arrays_loaded=False,
                scores_are_evaluation=False, runtime_fallback_used=False)
            immutable_json(path, r)
            results.append(r)
            beat(state="view_replayed" if verify else "view_complete", view=key, seconds=time.monotonic()-started)
    q = [v for r in results for v in r["queries"]]
    analysis = dict(manifest_sha256=digest,
        views=[dict(view=r["view"], path=str((root/"probes"/(r["view"]+".json")).relative_to(ROOT)),
            sha256=sha256(root/"probes"/(r["view"]+".json")), source_queries=len(r["queries"])) for r in results],
        source_queries=99, model_head_seed_views=len(results), source_query_view_instances=len(q),
        observed_agent_instances=sum(r["visible_agents"] for r in q),
        target_agent_instances=sum(r["neural_targets"] for r in q),
        unknown_CV_context_instances=sum(r["unknown_CV_context"] for r in q),
        rejected_model_outputs=sum(r["rejected_model_outputs"] for r in q),
        selected_instances_by_arm={k:sum(r["switch_counts"][k] for r in q) for k in q[0]["switch_counts"]},
        half_count_unmatched_queries=sum(not r["budgets"]["half"]["matched"] for r in q),
        synthetic_future_perturbation_views=len(results), source_bindings=len(manifest["source_bindings"]),
        new_training=False, outcome_metrics="not_run_input_only_probes", reserved_source_inference="not_run",
        source_admission=False, risk_calibration="not_run", independent_confirmation="not_run",
        joint_chain_frozen=True, deployment=False, stage5c_executed=False, smc_enabled=False)
    immutable_json(public/"analysis.json", analysis)
    assert_current(manifest)
    if verify:
        immutable_json(public/"replay.json", dict(analysis_sha256=sha256(public/"analysis.json"),
            replayed_query_view_instances=len(q), all_checks_passed=True, result_source="cached_verified",
            independent_research_confirmation=False, reserved_source_rows=0))
    beat(state="replayed" if verify else "complete", analysis_sha256=sha256(public/"analysis.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("audit-only", "freeze", "run", "verify"):
        mode.add_argument("--"+name, action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, manifest, context, alignment = specification()
    public, root = ROOT/cfg["reports"], ROOT/cfg["output"]
    root.mkdir(parents=True, exist_ok=True)
    def beat(**v):
        row = dict(pid=os.getpid(), timestamp_unix=time.time(), **v)
        json_write(root/"heartbeat.json", row)
        with (root/"events.jsonl").open("a") as stream:
            stream.write(json.dumps(row)+"\n")
        print(json.dumps(row), flush=True)
    with (root/"runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.freeze:
            immutable_json(public/"policy_manifest.json", manifest)
            beat(state="chain_frozen_no_source_probe", manifest_sha256=sha256(public/"policy_manifest.json"))
            return
        scenes = source_inputs(manifest, context, alignment)
        if args.audit_only:
            beat(state="preflight_pass", source_queries=len(scenes), bindings=len(manifest["source_bindings"]),
                 future_target_arrays_loaded=False, reserved_source_rows=0)
            return
        if not (public/"policy_manifest.json").exists() or json.loads((public/"policy_manifest.json").read_text()) != manifest:
            raise ValueError("Complete chain must be frozen before model inference")
        run(cfg, manifest, scenes, root, public, beat, args.verify)


if __name__ == "__main__":
    main()
