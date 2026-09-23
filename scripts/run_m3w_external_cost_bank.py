"""Fit a fixed OOF gain/harm bank; no external data or policy readout."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

if platform.system() == "Darwin" and platform.machine() != "arm64":
    raise RuntimeError("Native arm64 .venv-pytorch required before numerical imports")
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import sklearn
import torch

from scripts.run_m3w_external_predictor_refit import load as predictor_load
from scripts.run_m3w_native_forecast import assert_current, array_hash, immutable_json, json_write
from src.evaluation.m3w_recording_lineage import sha256
from src.evaluation.m3w_native_metrics import native_errors
from src.evaluation.m3w_forecast_cost_bounds import disagreement, bounded_fractions
from src.training.m3w_external_cost_bank import assemble_oof, fraction_targets, fit_forest, predict_forest
from src.world_model.m3w_native_forecast import fold_design
from src.world_model.m3w_native_gain_harm import cost_features, preprocess
from src.world_model.m3w_bounded_cost_head import build, fit, predict

CONFIG = "configs/m3w_external_cost_bank_v1.json"
CODE = ("scripts/run_m3w_external_cost_bank.py", "src/training/m3w_external_cost_bank.py",
        "tests/test_m3w_external_cost_bank.py", "src/world_model/m3w_bounded_cost_head.py",
        "src/world_model/m3w_native_gain_harm.py", "src/evaluation/m3w_forecast_cost_bounds.py")


def load():
    cfg = json.loads((ROOT/CONFIG).read_text())
    pcfg, data, _, pid = predictor_load()
    old_neural = json.loads((ROOT/"configs/m3w_bounded_cost_v1.json").read_text())
    old_forest = json.loads((ROOT/"configs/m3w_forest_cost_v1.json").read_text())
    if (cfg["sites"] != pcfg["sites"] or cfg["seeds"] != pcfg["seeds"]
            or cfg["families"] != pcfg["families"] or sklearn.__version__ != "1.8.0"
            or cfg["heads"] != ["bounded_fraction", "matched_fraction_forest"]
            or cfg["training"] != old_neural["training"] or cfg["forest"] != old_forest["forest"]
            or cfg["feature_dim"] != 356 or any(cfg[k] for k in (
                "future_labels_input", "full_source_in_sample_cost_targets", "threshold_search",
                "model_selection", "reserved_source_readout", "independent_calibration",
                "independent_confirmation", "deployment", "stage5c_executed", "smc_enabled"))):
        raise ValueError("Fixed source-only matched cost-head registration required")
    bindings = dict(pid["source_bindings"])
    def bind(path, expected=None):
        path = str(path)
        actual = sha256(ROOT/path)
        if (expected is not None and actual != expected) or (path in bindings and bindings[path] != actual):
            raise ValueError("Changed dependency: "+path)
        bindings[path] = actual
        return actual
    for path in (CONFIG, cfg["registration"], *CODE,
                 "configs/m3w_bounded_cost_v1.json", "configs/m3w_forest_cost_v1.json"):
        bind(path)
    full = cfg["source_predictor_analysis"]
    bind(full["path"], full["sha256"])
    pa = json.loads((ROOT/full["path"]).read_text())
    replay_path = str(Path(full["path"]).with_name("replay.json"))
    bind(replay_path)
    replay = json.loads((ROOT/replay_path).read_text())
    if (pa["identity"] != pid or pa["model_count"] != 6 or not replay["all_checks_passed"]
            or replay["analysis_sha256"] != full["sha256"]):
        raise ValueError("Full-source predictor bank must already be completed and replayed")
    for record in pa["models"]:
        bind(record["checkpoint"], record["checkpoint_sha256"])
    views = {}
    for family in cfg["families"]:
        info = cfg["oof_analyses"][family]
        bind(info["path"], info["sha256"])
        bind(info["replay"])
        report = json.loads((ROOT/info["path"]).read_text())
        proof = json.loads((ROOT/info["replay"]).read_text())
        if not proof["all_checks_passed"] or proof["analysis_sha256"] != info["sha256"]:
            raise ValueError("Unverified OOF prediction cache")
        if report["identity"]["population_sha256"] != pid["source_population_sha256"]:
            raise ValueError("OOF source population differs")
        for path, digest in report["identity"]["source_bindings"].items():
            bind(path, digest)
        field = "trial" if family == "transformer" else "view"
        preds = {r[field]: r for r in report["predictions" if family == "transformer" else "archives"]}
        states = {r[field]: r for r in report["training"]}
        for seed in cfg["seeds"]:
            producers = []
            for site in cfg["sites"]:
                key = f"{site}_native_coordinate_seed{seed}" if family == "transformer" else f"{site}_seed{seed}"
                record, archive = states[key], preds[key]
                bind(record["checkpoint"], record["checkpoint_sha256"])
                bind(archive["path"], archive["sha256"])
                cp = torch.load(ROOT/record["checkpoint"], map_location="cpu", weights_only=False)
                ti = cp["identity"]
                fold = fold_design(data, site, "native_coordinate")
                if (ti["identity"] != report["identity"] or ti["held_site"] != site
                        or ti.get("seed", cp["seed"]) != seed or cp["seed"] != seed
                        or cp["step"] != pcfg["training"]["steps"] or cp["settings"] != pcfg["training"]
                        or ti["normalizers"] != fold["normalizers"]
                        or ti["train_ids_sha256"] != array_hash(fold["train_ids"])
                        or ti["held_ids_sha256"] != array_hash(fold["held_ids"])
                        or ti["factors_sha256"] != array_hash(fold["factors"])
                        or int(cp["draws"].sum()) != 256000 or cp["draws"][fold["held_ids"]].any()):
                    raise ValueError("OOF producer includes held rows or mismatches the frozen fit")
                np.testing.assert_array_equal(cp["train_ids"], fold["train_ids"])
                np.testing.assert_array_equal(cp["factors"], fold["factors"])
                if sorted(ti["normalizers"]) != sorted(set(cfg["sites"])-{site}):
                    raise ValueError("Held site entered producer normalization")
                with np.load(ROOT/archive["path"], allow_pickle=False) as z:
                    if set(z.files) != {"ids", "prediction"}:
                        raise ValueError("Unexpected prediction cache fields")
                    np.testing.assert_array_equal(z["ids"], fold["held_ids"])
                    if z["prediction"].shape != (len(fold["held_ids"]), 12, 2) or not np.isfinite(z["prediction"]).all():
                        raise ValueError("Invalid OOF prediction")
                producers.append(dict(row_site=site, family=family, seed=seed,
                    training_sites=sorted(ti["normalizers"]), prediction=archive,
                    checkpoint=record["checkpoint"], checkpoint_sha256=record["checkpoint_sha256"],
                    train_ids_sha256=array_hash(cp["train_ids"]), held_ids_sha256=array_hash(fold["held_ids"]),
                    held_rows_sampled=0, input_normalizers_exclude_row_site=True))
            views[f"{family}_seed{seed}"] = dict(family=family, seed=seed, producers=producers)
    identity = dict(config=cfg, source_bindings=bindings, torch=torch.__version__, sklearn=sklearn.__version__,
        numpy=np.__version__, architecture=platform.machine(), population_sha256=pid["source_population_sha256"],
        source_rows=len(data["sites"]), views=views, full_source_checkpoint_costs_used_for_training=False)
    assert_current(identity)
    return cfg, data, identity


def training_arrays(data, view):
    parts = []
    for producer in view["producers"]:
        with np.load(ROOT/producer["prediction"]["path"], allow_pickle=False) as z:
            parts.append(dict(producer, ids=z["ids"].copy(), prediction=z["prediction"].copy()))
    p = assemble_oof(data["sites"], parts, seed=view["seed"], family=view["family"])
    b = data["geometry"][:, 332:356].reshape(-1, 12, 2)
    x, _ = cost_features(data["geometry"], p, data["scale"])
    d = disagreement(p, b, data["scale"]).mean(1)
    x = np.column_stack((x, np.log1p(d))).astype(np.float32)
    cv, _ = native_errors(b, data["target"], data["valid"], data["scale"])
    error, _ = native_errors(p, data["target"], data["valid"], data["scale"])
    delta = cv-error
    y = np.column_stack((np.maximum(delta, 0), np.maximum(-delta, 0)))
    full = data["valid"].all(1)
    bounded_fractions(y, d, full)
    y[~full], cv[~full] = np.nan, np.nan
    pr = preprocess(x, y, cv, data["sites"], "external_reserved_not_a_training_site")
    if x.shape != (len(data["sites"]), 356) or not np.array_equal(pr["known"], full):
        raise ValueError("Mismatched causal feature schema or full-grid supervision")
    return x, y, d, pr


def trial_id(identity, key, x, y, d, pr):
    return dict(identity=identity, view=key, inputs_sha256=array_hash(x, d),
        labels_sha256=array_hash(y), support_sha256=array_hash(pr["known"]),
        preprocessing_sha256=array_hash(pr["mean"], pr["std"], pr["weights"]))


def checked_receipt(folder, ti, *, forest):
    record = json.loads((folder/"complete.json").read_text())
    if (record["identity"] != ti or not record["fit"]["complete"]
            or sha256(ROOT/record["checkpoint"]) != record["checkpoint_sha256"]
            or record["fit"]["trees" if forest else "step"] != (128 if forest else 3000)):
        raise ValueError("Missing, changed or incomplete cost-head endpoint")
    return record


def run(cfg, data, identity, args, beat):
    reports = []
    root = ROOT/cfg["output"]
    if args.verify:
        for key in identity["views"]:
            for head in cfg["heads"]:
                if not (root/"trials"/key/head/"complete.json").exists():
                    raise ValueError("Replay cannot create a missing fit")
    for key, view in identity["views"].items():
        if args.trial and key != args.trial:
            continue
        x, y, d, pr = training_arrays(data, view)
        ti = trial_id(identity, key, x, y, d, pr)
        states = {}
        for head in cfg["heads"]:
            forest = head == "matched_fraction_forest"
            folder = root/"trials"/key/head
            if (folder/"complete.json").exists():
                record = checked_receipt(folder, ti, forest=forest)
                beat(state="cached_verified_complete", trial=key, head=head)
            else:
                if args.verify:
                    raise ValueError("No new training permitted in verification")
                beat(state="training_started", trial=key, head=head, complete_rows=int(pr["known"].sum()))
                if forest:
                    _, result = fit_forest(x, y, d, pr, states["bounded_fraction"]["draws"],
                        settings=cfg["forest"], seed=view["seed"], identity=ti, directory=folder,
                        heartbeat=lambda **v: beat(trial=key, head=head, **v),
                        resume=args.resume, stop_at=args.stop_at_trees)
                    checkpoint = folder/"checkpoint.joblib"
                else:
                    _, result = fit(x, y, d, data["sites"], pr, arm="bounded_fraction", seed=view["seed"],
                        settings=cfg["training"], identity=ti, directory=folder, resume=args.resume,
                        heartbeat=lambda **v: beat(trial=key, head=head, **v))
                    checkpoint = folder/"checkpoint.pt"
                if not result["complete"]:
                    beat(state="timing_prefix_complete_not_full", trial=key, head=head, fit=result)
                    return
                assert_current(identity)
                record = dict(identity=ti, head=head, fit=result, checkpoint=str(checkpoint.relative_to(ROOT)),
                    checkpoint_sha256=sha256(checkpoint))
                immutable_json(folder/"complete.json", record)
                beat(state="fit_complete", trial=key, head=head, seconds=result["seconds"])
            cp = joblib.load(ROOT/record["checkpoint"]) if forest else torch.load(
                ROOT/record["checkpoint"], map_location="cpu", weights_only=False)
            if (cp["identity"] != ti or cp["seed"] != view["seed"]
                    or cp["settings"] != cfg["forest" if forest else "training"]):
                raise ValueError("Checkpoint identity mismatch")
            for field in ("mean", "std", "known", "weights"):
                np.testing.assert_array_equal(cp["preprocess"][field], pr[field])
            if cp["preprocess"]["cost_scale"] != pr["cost_scale"]:
                raise ValueError("Changed cost scale")
            if (cp["draws"].shape != (len(x),) or cp["draws"].sum() != 768000
                    or cp["draws"][~pr["known"]].any()):
                raise ValueError("Wrong fitting population or sampling budget")
            states[head] = cp
            if forest:
                np.testing.assert_array_equal(cp["draws"], states["bounded_fraction"]["draws"])
                _, weight = fraction_targets(y, d, pr["known"], cp["draws"])
                np.testing.assert_array_equal(cp["sample_weight"], weight)
                if len(cp["model"].estimators_) != cfg["forest"]["trees"]:
                    raise ValueError("Wrong forest endpoint")
            elif (cp["step"] != 3000 or cp["settings"] != cfg["training"]
                    or not all(torch.isfinite(v).all() for v in cp["model"].values())):
                raise ValueError("Wrong or nonfinite neural endpoint")
            reports.append(dict(view=key, head=head, fit=record["fit"], checkpoint=record["checkpoint"],
                checkpoint_sha256=record["checkpoint_sha256"], draws_sha256=array_hash(cp["draws"]),
                rows=len(x), complete_rows=int(pr["known"].sum()),
                unknown_rows_sampled=int(cp["draws"][~pr["known"]].sum()),
                inputs_sha256=ti["inputs_sha256"], labels_sha256=ti["labels_sha256"],
                preprocessing_sha256=ti["preprocessing_sha256"], score_source="not_yet_probed"))
        beat(state="view_fits_complete", trial=key)
    if args.trial:
        beat(state="single_registered_view_complete_not_whole_matrix")
        return
    # Only after the whole matrix is fitted do any fixed probe inferences occur.
    for key, view in identity["views"].items():
        x, y, d, pr = training_arrays(data, view)
        ti = trial_id(identity, key, x, y, d, pr)
        ids = np.concatenate([g[np.linspace(0, len(g)-1, 16, dtype=int)]
            for g in [np.flatnonzero(data["sites"] == site) for site in cfg["sites"]]])
        for head in cfg["heads"]:
            record = next(r for r in reports if r["view"] == key and r["head"] == head)
            forest = head == "matched_fraction_forest"
            cp = joblib.load(ROOT/record["checkpoint"]) if forest else torch.load(
                ROOT/record["checkpoint"], map_location="cpu", weights_only=False)
            if forest:
                score = predict_forest(cp["model"], x[ids], d[ids], pr)
            else:
                model = build(x.shape[1], cfg["training"]["width"], view["seed"])
                model.load_state_dict(cp["model"])
                score = predict(model, x[ids], d[ids], pr, "bounded_fraction")
            if (score.shape != (len(ids), 2) or not np.isfinite(score).all()
                    or (score < 0).any() or np.any(score.sum(1) > d[ids]+2e-6*(1+d[ids]))):
                raise ValueError("Invalid bounded cost output")
            record.update(score_source="fixed_source_input_probe_not_evaluation", probe_rows=len(ids),
                probe_ids_sha256=array_hash(ids), score_sha256=array_hash(score), cost_bound_checked=True)
        beat(state="input_probe_replayed" if args.verify else "input_probe_complete", trial=key)
    analysis = dict(identity=identity, models=reports, trained_neural_heads=6, trained_forests=6,
        source_population=len(data["sites"]), source_sites=cfg["sites"], oof_predictor_count=24,
        cost_target_rows_repeated_over_families_seeds=len(data["sites"])*6,
        neural_optimizer_updates=sum(r["fit"]["step"] for r in reports if r["head"] == "bounded_fraction"),
        neural_sampler_draws=sum(r["fit"]["total_draws"] for r in reports if r["head"] == "bounded_fraction"),
        summed_fit_seconds=sum(r["fit"]["seconds"] for r in reports),
        source_probe_cost_rows=sum(r["probe_rows"] for r in reports),
        full_source_in_sample_cost_labels=False, risk_head_heldout_evaluation="not_run",
        matching_neural_forest_sample_counts=True, matching_fraction_squared_error_objective=True,
        independent_calibration="not_run", independent_confirmation="not_run",
        reserved_source_inference="not_run", joint_controller_frozen=False,
        deployment=False, stage5c_executed=False, smc_enabled=False)
    path = ROOT/cfg["reports"]/"analysis.json"
    if args.verify and not path.exists():
        raise ValueError("Missing original complete analysis")
    immutable_json(path, analysis)
    assert_current(identity)
    if args.verify:
        immutable_json(path.with_name("replay.json"), dict(analysis_sha256=sha256(path),
            cost_heads_reloaded=12, fixed_source_cost_rows_replayed=768, matched_sampler_pairs=6,
            all_checks_passed=True, result_source="cached_verified", reserved_source_rows=0))
    beat(state="verified" if args.verify else "complete", analysis_sha256=sha256(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("audit-only", "resume", "verify"):
        parser.add_argument("--"+flag, action="store_true")
    parser.add_argument("--trial")
    parser.add_argument("--stop-at-trees", type=int)
    args = parser.parse_args()
    if ((args.audit_only and args.verify) or ((args.audit_only or args.verify) and
            (args.trial or args.stop_at_trees is not None)) or (args.stop_at_trees is not None and not args.trial)):
        parser.error("Choose a disjoint phase; timing prefix must name its fixed view")
    if args.stop_at_trees is not None and not 1 <= args.stop_at_trees <= 128:
        parser.error("Timing prefix must be within the fixed 128-tree budget")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cfg = json.loads((ROOT/CONFIG).read_text())
    root = ROOT/cfg["output"]
    root.mkdir(parents=True, exist_ok=True)
    if subprocess.run(["git", "check-ignore", "--quiet", str(root/"checkpoint.pt")], cwd=ROOT).returncode:
        raise ValueError("Large artifacts must remain Git ignored")
    def beat(**values):
        event = dict(pid=os.getpid(), timestamp_unix=time.time(), **values)
        json_write(root/"heartbeat.json", event)
        with (root/"events.jsonl").open("a") as stream:
            stream.write(json.dumps(event)+"\n")
        print(json.dumps(event), flush=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("Resume saved checkpoint")))
    with (root/"runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, data, identity = load()
        if args.trial and args.trial not in identity["views"]:
            raise ValueError("Unregistered trial")
        immutable_json(root/"identity.json", identity)
        if args.audit_only:
            beat(state="preflight_pass", bindings=len(identity["source_bindings"]),
                source_rows=len(data["sites"]), verified_oof_producers=24, expected_new_heads=12,
                reserved_source_rows=0)
        else:
            run(cfg, data, identity, args, beat)


if __name__ == "__main__":
    main()
