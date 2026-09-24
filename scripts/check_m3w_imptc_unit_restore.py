"""Separate-coordinate arithmetic check, without opening prediction targets."""
import json
from pathlib import Path
import platform
import sys

if platform.system() == "Darwin" and platform.machine() != "arm64":
    raise RuntimeError("Native arm64 runtime required")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.evaluation.m3w_experiment_contract import file_digest
from src.data_unification.m3w_external_prefix_adapter import ExternalPrefixAdapter
from src.data_unification.m3w_unit_free_prefix import UnitFreePrefixAdapter
from scripts.run_m3w_native_forecast import immutable_json


def read(path):
    return json.loads((ROOT/path).read_text())


def main():
    cfg = read("configs/m3w_imptc_input_contract_v1.json")
    out, report = ROOT/cfg["output"], ROOT/cfg["reports"]
    identity = read(out/"identity.json")
    for path, digest in identity["code_hashes"].items():
        if file_digest(ROOT/path) != digest:
            raise ValueError("Changed input producer code")
    analysis = read(report/"analysis.json")
    private = analysis["private_arrays"]
    if file_digest(ROOT/private["path"]) != private["sha256"]:
        raise ValueError("Changed private probe arrays")
    support = read(out/"prefix_selection.json")
    audit = read(cfg["source_audit"])
    if file_digest(ROOT/cfg["source_audit"]) != cfg["source_audit_sha256"]:
        raise ValueError("Changed source audit")
    for item in audit["cache_files"]:
        if file_digest(ROOT/item["path"]) != item["sha256"]:
            raise ValueError("Changed verified source cache")
    restored = {(mode, factor, model): [] for mode in ("legacy", "unit_free")
                for factor in (.01, 1., 100.) for model in cfg["cost_actions"]}
    cache, start, checks = {}, 0, 0
    with np.load(ROOT/private["path"], allow_pickle=False) as arrays:
        for query in support["queries"]:
            seq, frame = query["sequence"], query["frame"]
            if seq not in cache:
                cache[seq] = np.load(ROOT/f"data/stage_cvpr2027_experiments/imptc_intake_v1/{seq}_rows.npy",
                                     allow_pickle=False, mmap_mode="r")
            rows = cache[seq]
            b = rows[(rows["frame_id"] >= frame-7) & (rows["frame_id"] <= frame)]
            p = np.column_stack((b["frame_id"], b["agent_id"], b["x"], b["y"]))
            end = start + len(query["agents"])
            for mode, cls in (("legacy", ExternalPrefixAdapter), ("unit_free", UnitFreePrefixAdapter)):
                for factor in (.01, 1., 100.):
                    points = p.copy()
                    points[:, 2:] *= factor
                    reader = cls(points, query_frame=frame, recording_id=seq)
                    geometry, scene = reader.geometry_batch()
                    tag = mode + "_" + str(factor).replace(".", "p")
                    np.testing.assert_array_equal(geometry, arrays[tag+"_geometry"][start:end])
                    assert [a["agent_id"] for a in scene["agents"]] == query["agents"]
                    for model in cfg["cost_actions"]:
                        prediction = arrays[tag+"_"+model+"_prediction"][start:end]
                        rows_restored = []
                        for a, forecast in zip(scene["agents"], prediction):
                            t = a["coordinate_transform"]
                            # Explicit scalar arithmetic, separate from restore_scene_rollouts.
                            local = forecast.astype(float)*t["scale"]
                            x = local[:, 0]*t["rotation"][0, 0] + local[:, 1]*t["rotation"][0, 1] + t["origin_xy"][0]
                            y = local[:, 0]*t["rotation"][1, 0] + local[:, 1]*t["rotation"][1, 1] + t["origin_xy"][1]
                            xy = np.column_stack((x, y))
                            if mode == "unit_free":
                                xy = xy*reader.outer_scale + reader.outer_origin
                            xy /= factor
                            rows_restored.append(xy)
                            if model == "damped_velocity_005" and mode == "legacy":
                                own = p[p[:, 1] == a["agent_id"]]
                                own = own[np.argsort(own[:, 0])]
                                v = own[-1, 2:] - own[-2, 2:]
                                expected = own[-1, 2:] + (-np.expm1(-.05*np.arange(1, 13))/.05)[:, None]*v
                                np.testing.assert_allclose(xy, expected, rtol=1e-5, atol=1e-6)
                                checks += 1
                        restored[mode, factor, model].append(np.array(rows_restored))
            start = end
    contrasts = {}
    for (mode, factor, model), parts in restored.items():
        if factor == 1.:
            continue
        a = np.concatenate(parts)
        b = np.concatenate(restored[mode, 1., model])
        distance = np.sqrt(((a-b)**2).sum(-1)).max(1)
        contrasts[f"{mode}:{factor}:{model}"] = dict(
            maximum_source_coordinate_difference=float(distance.max()),
            p99_source_coordinate_difference=float(np.quantile(distance, .99)),
            mean_max_difference=float(distance.mean()),
            windows_over_1e_5=int((distance > 1e-5).sum()))
    immutable_json(report/"restored_coordinate_check.json", dict(
        result_source="fresh_run", method="separate_arithmetic_same_agent_not_independent_review",
        code_sha256=file_digest(Path(__file__)),
        input_analysis_sha256=file_digest(report/"analysis.json"),
        windows=start, explicit_damping_checks=checks, contrasts=contrasts,
        outcome_readout=False, predictive_improvement_claim=False,
        limitation="Comparison of two forecasts on identical past under unit changes, not error against future truth"))
    print(json.dumps(dict(windows=start, explicit_damping_checks=checks, status="pass")))


if __name__ == "__main__":
    main()
