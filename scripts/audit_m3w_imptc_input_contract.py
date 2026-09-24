"""Frozen-model numerical unit probes; no external forecasting error readout."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == "Darwin" and platform.machine() != "arm64":
    raise RuntimeError("Native arm64 runtime required before Torch import")
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key, "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import torch

from src.evaluation.m3w_experiment_contract import file_digest
from src.data_unification.m3w_external_prefix_adapter import ExternalPrefixAdapter
from src.data_unification.m3w_unit_free_prefix import UnitFreePrefixAdapter, unit_free_cost_features, SCHEMA
from src.world_model.m3w_native_forecast import pack_geometry
from src.world_model.m3w_native_gain_harm import cost_features
from src.world_model.m3w_supervised_intervention import build_forecaster
from src.training.m3w_easy_moment import predict as head_predict
from scripts.run_m3w_native_forecast import immutable_json, array_hash, json_write
from scripts.run_m3w_native_nested import write_arrays

CONFIG = "configs/m3w_imptc_input_contract_v1.json"
CODE = ("scripts/audit_m3w_imptc_input_contract.py",
        "src/data_unification/m3w_unit_free_prefix.py",
        "tests/test_m3w_unit_free_prefix.py",
        "src/data_unification/m3w_external_prefix_adapter.py",
        "src/data_unification/m3w_causal_recordings.py",
        "src/world_model/m3w_offline_visual_forecast.py")
SITES = {"coupa", "deathCircle", "gates", "hyang"}


def read(path):
    return json.loads((ROOT/path).read_text())


def beat(**values):
    print(json.dumps(dict(pid=os.getpid(), **values)), flush=True)


def check(path, digest):
    if file_digest(ROOT/path) != digest:
        raise ValueError("Changed bound artifact: " + path)


def verify_lineage(cfg):
    identity = read(cfg["producer_identity"])
    bindings = identity["source_bindings"]
    checked_bytes = 0
    for i, (path, sha) in enumerate(bindings.items()):
        check(path, sha)
        checked_bytes += (ROOT/path).stat().st_size
        if i % 500 == 0:
            beat(state="hash_verification_no_numeric_outcome_readout", files=i, total=len(bindings))
    manifest_path = "data/stage_cvpr2027_experiments/sdd_auxiliary_v1/inputs/manifest.json"
    check(manifest_path, bindings[manifest_path])
    manifest = read(manifest_path)
    records = [r for r in manifest["records"] if r["recording"].split("/")[0] in SITES]
    if (len(records) != 33 or sum(r["rows"] for r in records) != 175756
            or any(r["original_split"] != "train" or r["data_role"] != "supervised_auxiliary_training" for r in records)):
        raise ValueError("Changed approved producer population")
    receipts = []
    for family, folder in (("transformer", "native_forecast_v1"),
                           ("eqmotion", "native_eqmotion_v1"), ("moments", "net_easy_moment_v1")):
        found = []
        for path in sorted((ROOT/"data/stage_cvpr2027_experiments"/folder/"trials").rglob("complete.json")):
            r = json.loads(path.read_text())
            if family == "transformer" and r["identity"]["objective"] != "native_coordinate":
                continue
            check(r["checkpoint"], r["checkpoint_sha256"])
            ti = r["identity"]
            held = ti["view"].split("_seed")[0] if family == "moments" else ti["held_site"]
            sites = set(ti["training_sites"] if family == "moments" else ti["normalizers"])
            if (held not in SITES or sites != SITES - {held} or not r["fit"]["complete"]
                    or (family == "moments" and (r["fit"]["trees"] != 128 or r["fit"]["unknown_sampled"] != 0))
                    or (family != "moments" and (r["fit"]["step"] != 4000 or r["fit"]["held_rows_sampled"] != 0))):
                raise ValueError("Producer exclusion or fit receipt invalid")
            found.append(dict(family=family, receipt=str(path.relative_to(ROOT)),
                receipt_sha256=file_digest(path), checkpoint=r["checkpoint"],
                checkpoint_sha256=r["checkpoint_sha256"], training_sites=sorted(sites), held_site=held))
        if len(found) != (36 if family == "moments" else 12):
            raise ValueError("Missing frozen producers")
        receipts.extend(found)
    external = sorted(k for k in bindings if k.startswith("external_data/"))
    if not all(k.startswith("external_data/StanfordDroneDataset/annotations/") for k in external):
        raise ValueError("Unreviewed external raw source in current producer chain")
    return dict(result_source="cached_verified", bound_files=len(bindings), bound_bytes=checked_bytes,
        producer_identity_sha256=file_digest(ROOT/cfg["producer_identity"]),
        records=33, fitting_windows=175756, source_sites=sorted(SITES),
        manifest_sha256=file_digest(ROOT/manifest_path),
        bound_external_raw_sources=external, receipts=receipts,
        imptc_or_vru_raw_path_in_verified_local_chain=False,
        remote_exposure_checked=False, project_wide_scientific_independence_established=False,
        limitation="Local fitted producer chain only; byte hashing is not a project-wide exposure proof")


def load_models(lineage):
    models, heads = {}, {}
    for family in ("transformer", "eqmotion"):
        part = "coupa_native_coordinate_seed17" if family == "transformer" else "coupa_seed17"
        r = next(r for r in lineage["receipts"] if r["family"] == family and part in r["receipt"])
        receipt = read(r["receipt"])
        cp = torch.load(ROOT/r["checkpoint"], map_location="cpu", weights_only=False)
        if cp["identity"] != receipt["identity"] or cp["step"] != 4000:
            raise ValueError("Checkpoint/receipt identity differs")
        cfg = read("configs/m3w_native_forecast_v1.json" if family == "transformer" else "configs/m3w_native_eqmotion_v1.json")
        model = build_forecaster(cfg["architecture"])
        model.load_state_dict(cp["model"])
        model.eval()
        models[family] = model
    for action in ("damped_velocity_005", "transformer", "eqmotion"):
        r = next(r for r in lineage["receipts"] if r["family"] == "moments" and f"coupa_seed17/{action}/" in r["receipt"])
        cp = joblib.load(ROOT/r["checkpoint"])
        if cp["identity"] != read(r["receipt"])["identity"] or cp["seed"] != 17:
            raise ValueError("Moment checkpoint/receipt identity differs")
        heads[action] = cp
    return models, heads


def collect(cfg, audit):
    variants = {(mode, factor): {"geometry": [], "scale": []}
                for mode in ("legacy", "unit_free") for factor in cfg["coordinate_factors"]}
    queries, prefix_checks, no_target, degenerate = [], 0, 0, 0
    for item in audit["cache_files"]:
        check(item["path"], item["sha256"])
    for seq in audit["sequences"]:
        path = f"data/stage_cvpr2027_experiments/imptc_intake_v1/{seq['sequence']}_rows.npy"
        rows = np.load(ROOT/path, mmap_mode="r", allow_pickle=False)
        frames = np.unique(rows["frame_id"])
        frames = frames[(frames >= 7) & (frames % cfg["query_modulus"] == cfg["query_remainder"])]
        for q in frames:
            # The only numerically read coordinates are these past eight indices.
            use = (rows["frame_id"] >= q-7) & (rows["frame_id"] <= q)
            block = rows[use]
            p = np.column_stack((block["frame_id"], block["agent_id"], block["x"], block["y"]))
            if not len(p) or not (p[:, 0] == q).any():
                continue
            case = {}
            for mode, cls in (("legacy", ExternalPrefixAdapter), ("unit_free", UnitFreePrefixAdapter)):
                for factor in cfg["coordinate_factors"]:
                    changed = p.copy()
                    changed[:, 2:] *= factor
                    try:
                        adapter = cls(changed, query_frame=int(q), recording_id=seq["sequence"])
                    except ValueError as e:
                        if mode != "unit_free" or "Zero-extent" not in str(e):
                            raise
                        case = None
                        break
                    g, scene = adapter.geometry_batch()
                    ids = np.array([a["agent_id"] for a in scene["agents"]], dtype=np.int64)
                    scales = np.array([a["coordinate_transform"]["scale"] for a in scene["agents"]])
                    case[mode, factor] = (g, scales, ids)
                if case is None:
                    break
            if case is None:
                degenerate += 1
                continue
            ids = case["legacy", 1.][2]
            if not len(ids):
                no_target += 1
                continue
            for key, (g, scales, other) in case.items():
                np.testing.assert_array_equal(ids, other)
                pack_geometry(g)
                variants[key]["geometry"].append(g)
                variants[key]["scale"].append(scales)
            # Exercise the source-prefix gate with an appended future-only agent.
            extended = np.r_[p, [[int(q)+1, 2**30, 9e10, -9e10]]]
            admitted = extended[(extended[:, 0] >= q-7) & (extended[:, 0] <= q)]
            g2, _ = UnitFreePrefixAdapter(admitted, query_frame=int(q), recording_id=seq["sequence"]).geometry_batch()
            np.testing.assert_array_equal(case["unit_free", 1.][0], g2)
            prefix_checks += 1
            queries.append(dict(sequence=seq["sequence"], frame=int(q), agents=ids.tolist()))
        beat(state="past_prefix_built", sequence=seq["sequence"], queries=len(queries))
    if not queries:
        raise ValueError("No eligible diagnostic queries")
    for v in variants.values():
        for key in v:
            v[key] = np.concatenate(v[key])
    return variants, dict(queries=queries, query_count=len(queries),
        target_windows=sum(len(q["agents"]) for q in queries), future_only_agent_checks=prefix_checks,
        unsupported_zero_extent_queries=degenerate, no_eligible_target_queries=no_target,
        future_labels_numerically_read=False, source_fulltrack_class_used=False)


def infer(model, geometry):
    with torch.inference_mode():
        values = [model(pack_geometry(geometry[s:s+128])).numpy() for s in range(0, len(geometry), 128)]
    p = np.concatenate(values)
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite numerical forward probe")
    return p


def native_features(g, p, scale):
    x, _ = cost_features(g, p, scale)
    d = np.linalg.norm(p.astype(float)-g[:, 332:356].reshape(-1, 12, 2), axis=-1).mean(1)*scale
    return np.column_stack((x, np.log1p(d))).astype(np.float32)


def compare(a, b):
    delta = np.abs(a.astype(float) - b.astype(float))
    row = delta.reshape(len(a), -1).max(1)
    return dict(max_absolute=float(row.max()), p99_absolute=float(np.quantile(row, .99)),
                rows_over_1e_5=int((row > 1e-5).sum()))


def probe(variants, models, heads):
    saved, result = {}, {}
    for mode in ("legacy", "unit_free"):
        outputs = {}
        for factor in (.01, 1., 100.):
            v = variants[mode, factor]
            g, scale = v["geometry"], v["scale"]
            predictions = {name: infer(model, g) for name, model in models.items()}
            predictions["damped_velocity_005"] = g[:, 356:380].reshape(-1, 12, 2).copy()
            fields = {"geometry": g, "scale": scale}
            for name, p in predictions.items():
                fields[name+"_prediction"] = p
                if mode == "legacy":
                    x = native_features(g, p, scale)
                    fields[name+"_cost_features"] = x
                    fields[name+"_moments"] = head_predict(heads[name]["model"], x, heads[name]["preprocess"])
                else:
                    fields[name+"_cost_features"] = unit_free_cost_features(g, p)[0]
            outputs[factor] = fields
            for name, a in fields.items():
                saved[f"{mode}_{str(factor).replace('.', 'p')}_{name}"] = a
            beat(state="frozen_forward_no_error_readout", mode=mode, factor=factor, rows=len(g))
        comparisons = {}
        for factor in (.01, 100.):
            base, other = outputs[1.], outputs[factor]
            row = {name: compare(base[name], other[name]) for name in base if name != "scale"}
            for name in heads:
                if mode == "legacy":
                    a, b = base[name+"_moments"], other[name+"_moments"]
                    row[name+"_net_gain_sign_changed"] = int(((a[:, 0] > a[:, 1]) != (b[:, 0] > b[:, 1])).sum())
            comparisons[str(factor)] = row
        result[mode] = comparisons
    return result, saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cfg = read(CONFIG)
    if (cfg["query_modulus"] != 181 or cfg["query_remainder"] != 7
            or cfg["coordinate_factors"] != [.01, 1., 100.]
            or cfg["source_index_stride"] != 1 or cfg["fixed_predictor_view"] != "coupa_seed17"
            or any(cfg[k] for k in ("future_label_readout", "model_fitting", "scientific_calibration",
                                   "confirmation", "deployment", "stage5c_executed", "smc_enabled"))):
        raise ValueError("Fixed input-only diagnostic scope changed")
    check(cfg["source_audit"], cfg["source_audit_sha256"])
    out, report = ROOT/cfg["output"], ROOT/cfg["reports"]
    lineage = verify_lineage(cfg)
    identity = dict(config=cfg, code_hashes={p:file_digest(ROOT/p) for p in (CONFIG, cfg["registration"], *CODE)},
                    producer_identity_sha256=lineage["producer_identity_sha256"],
                    torch=torch.__version__, numpy=np.__version__, architecture=platform.machine(),
                    threads=4, interop_threads=1, num_workers=0)
    if args.verify and not (report/"analysis.json").exists():
        raise ValueError("Cannot verify a missing run")
    immutable_json(out/"identity.json", identity)
    immutable_json(report/"producer_chain.json", lineage)
    variants, support = collect(cfg, read(cfg["source_audit"]))
    immutable_json(out/"prefix_selection.json", support)
    models, heads = load_models(lineage)
    comparisons, arrays = probe(variants, models, heads)
    write_arrays(out/"unit_probes.npz", arrays)
    compatible = all(v["rows_over_1e_5"] == 0 for rows in comparisons["unit_free"].values()
                     for key, v in rows.items() if key.endswith("_cost_features") or key.endswith("_prediction"))
    analysis = dict(result_source="fresh_run", scope=cfg["scope"], identity_sha256=file_digest(out/"identity.json"),
        producer_chain_sha256=file_digest(report/"producer_chain.json"),
        source_audit_sha256=cfg["source_audit_sha256"], query_count=support["query_count"],
        target_windows=support["target_windows"], future_only_agent_checks=support["future_only_agent_checks"],
        degenerate_queries=support["unsupported_zero_extent_queries"], comparisons=comparisons,
        unit_free_measured_tolerance_pass=compatible,
        private_arrays=dict(path=str((out/"unit_probes.npz").relative_to(ROOT)), sha256=file_digest(out/"unit_probes.npz")),
        new_cost_schema=SCHEMA, old_head_compatible=False, new_head_refit="not_run",
        numerical_frozen_forward="fresh_run", forecasting_error_readout="not_run",
        source_status="quarantined_covariates_design_exposed_outcomes_unread",
        scientific_source_admission=False, matched_seconds=False, metric_claim=False,
        independent_confirmation=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False)
    immutable_json(report/"analysis.json", analysis)
    if args.verify:
        immutable_json(report/"verification.json", dict(result_source="cached_verified", exact_arrays=True,
            exact_analysis=True, analysis_sha256=file_digest(report/"analysis.json")))
    path = report/("verification_execution.json" if args.verify else "execution.json")
    if not path.exists():
        json_write(path, dict(result_source="fresh_run", pid=os.getpid(), elapsed_seconds=time.monotonic()-started,
            runtime=dict(architecture=platform.machine(), torch_threads=4, interop_threads=1, num_workers=0),
            new_optimizer_updates=0, future_error_readout=False))
    beat(state="verified" if args.verify else "complete", unit_free_measured_tolerance_pass=compatible,
         queries=support["query_count"], rows=support["target_windows"], seconds=time.monotonic()-started)


if __name__ == "__main__":
    main()
