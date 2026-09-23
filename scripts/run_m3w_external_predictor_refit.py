"""Fixed source-only predictors; reserved datasets never enter this runner."""
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
    raise RuntimeError("Native arm64 .venv-pytorch required before Torch import")
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.run_m3w_native_forecast import (
    load as source_load, array_hash, assert_current, completed, immutable_json, json_write,
)
from src.evaluation.m3w_recording_lineage import sha256
from src.training.m3w_frozen_source_refit import full_source_design
from src.world_model.m3w_native_forecast import fit_trial, predict
from src.world_model.m3w_supervised_intervention import build_forecaster
from src.world_model.m3w_eqmotion_adapter import verified_source


CONFIG = "configs/m3w_external_predictor_refit_v1.json"
CODE = (
    "scripts/run_m3w_external_predictor_refit.py", "src/training/m3w_frozen_source_refit.py",
    "tests/test_m3w_frozen_source_refit.py", "src/world_model/m3w_eqmotion_adapter.py",
    "configs/m3w_eqmotion_source.json",
)


def load():
    cfg = json.loads((ROOT / CONFIG).read_text())
    if (cfg["sites"] != ["coupa", "deathCircle", "gates", "hyang"]
            or cfg["seeds"] != [17, 29, 43] or cfg["families"] != ["transformer", "eqmotion"]
            or cfg["objective"] != "native_coordinate" or any(cfg[key] for key in (
                "model_selection", "threshold_search", "reserved_inputs_read", "reserved_labels_read",
                "independent_calibration", "independent_confirmation", "deployment",
                "stage5c_execution", "smc_enabled"))):
        raise ValueError("Only the fixed source-only six-model refit is registered")
    reservation = cfg["reserved_sources"]
    if sha256(ROOT / reservation["path"]) != reservation["sha256"]:
        raise ValueError("Reserved source roles changed")
    parent, data, rows, identity = source_load()
    if (cfg["training"] != parent["training"] or cfg["runtime"] != parent["runtime"]
            or cfg["sites"] != parent["sites"] or cfg["seeds"] != parent["seeds"]):
        raise ValueError("Changed established budget, runtime or source population")
    bindings = dict(identity["source_bindings"])
    architectures = {}
    for family, path in cfg["architecture_sources"].items():
        original = json.loads((ROOT / path).read_text())
        if original["training"] != cfg["training"]:
            raise ValueError("Predictor families have different fitting budgets")
        architectures[family] = original["architecture"]
        bindings[path] = sha256(ROOT / path)
    _, author = verified_source()
    source_spec = json.loads((ROOT / "configs/m3w_eqmotion_source.json").read_text())
    for path, digest in author["files_sha256"].items():
        bindings[str(Path(source_spec["destination"]) / path)] = digest
    for path in (CONFIG, cfg["registration"], reservation["path"], *CODE):
        bindings[path] = sha256(ROOT / path)
    fold = full_source_design(data, cfg["sites"])
    full_identity = dict(config=cfg, architectures=architectures, author=author,
        source_bindings=bindings, source_population_sha256=identity["population_sha256"],
        torch=torch.__version__, numpy=np.__version__, architecture=platform.machine(),
        train_ids_sha256=array_hash(fold["train_ids"]), factors_sha256=array_hash(fold["factors"]),
        normalizers=fold["normalizers"], training_rows=len(data["sites"]),
        training_recordings=sorted(set(data["recordings"].tolist())),
        excluded_from_all_fitting=["DUT", "DroneCrowd", "SDD_original_validation_test", "bookstore"],
        independent_test_rows=0, out_of_fold_producer=False)
    assert_current(full_identity)
    return cfg, data, fold, full_identity


def trial_identity(identity, family, seed):
    return dict(identity=identity, family=family, seed=seed, scope="full_source_fit_no_held_readout")


def inspect_endpoint(cfg, data, fold, identity, family, seed):
    key = f"{family}_seed{seed}"
    directory = ROOT / cfg["output"] / "trials" / key
    ti = trial_identity(identity, family, seed)
    receipt = completed(directory / "complete.json", ti, cfg["training"])
    checkpoint = torch.load(ROOT / receipt["checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint["identity"] != ti or checkpoint["step"] != cfg["training"]["steps"]:
        raise ValueError("Checkpoint is not the registered final endpoint")
    np.testing.assert_array_equal(checkpoint["train_ids"], fold["train_ids"])
    np.testing.assert_array_equal(checkpoint["factors"], fold["factors"])
    expected_draws = cfg["training"]["steps"] * cfg["training"]["batch_size"]
    if (checkpoint["draws"].shape != (len(data["sites"]),)
            or int(checkpoint["draws"].sum()) != expected_draws
            or not checkpoint["losses"] or checkpoint["losses"][-1]["step"] != cfg["training"]["steps"]
            or not all(np.isfinite(row["loss"]) for row in checkpoint["losses"])
            or not all(torch.isfinite(value).all() for value in checkpoint["model"].values())):
        raise ValueError("Incomplete, nonfinite or mismatched training state")
    torch.manual_seed(seed)
    model = build_forecaster(identity["architectures"][family])
    model.load_state_dict(checkpoint["model"])
    ids = np.concatenate([group[np.linspace(0, len(group)-1, 16, dtype=int)] for group in fold["groups"]])
    prediction = predict(model, data, ids)
    if prediction.shape != (len(ids), 12, 2) or not np.isfinite(prediction).all():
        raise ValueError("Invalid source-only inference interface")
    report = dict(family=family, seed=seed, checkpoint=receipt["checkpoint"],
        checkpoint_sha256=receipt["checkpoint_sha256"], fit=receipt["fit"],
        source_input_probe_rows=len(ids), source_input_probe_ids_sha256=array_hash(ids),
        source_prediction_sha256=array_hash(prediction), draws_sha256=array_hash(checkpoint["draws"]),
        train_ids_sha256=array_hash(checkpoint["train_ids"]), factors_sha256=array_hash(checkpoint["factors"]),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        probe_error_scored=False, reserved_source_rows=0)
    return report


def verify_all(cfg, data, fold, identity, beat, verify):
    # Require all final receipts before checking model inference; no pilot readout.
    for family in cfg["families"]:
        for seed in cfg["seeds"]:
            completed(ROOT / cfg["output"] / "trials" / f"{family}_seed{seed}" / "complete.json",
                      trial_identity(identity, family, seed), cfg["training"])
    reports = []
    for family in cfg["families"]:
        for seed in cfg["seeds"]:
            beat(state="verifying_endpoint", family=family, seed=seed)
            reports.append(inspect_endpoint(cfg, data, fold, identity, family, seed))
    for seed in cfg["seeds"]:
        a, b = [r for r in reports if r["seed"] == seed]
        if any(a[field] != b[field] for field in ("draws_sha256", "train_ids_sha256", "factors_sha256")):
            raise ValueError("Unmatched per-seed family sampling or preprocessing")
    analysis = dict(scope="fixed_source_predictor_refit_not_external_results", identity=identity,
        models=reports, model_count=len(reports), source_rows=len(data["sites"]),
        source_sites=cfg["sites"], optimizer_updates=sum(r["fit"]["step"] for r in reports),
        total_draws=sum(r["fit"]["total_draws"] for r in reports),
        summed_fit_seconds=sum(r["fit"]["seconds"] for r in reports),
        fixed_source_probe_predictions=sum(r["source_input_probe_rows"] for r in reports),
        matching_families_per_seed=True, no_model_selection=True,
        source_training_only=True, cost_heads_refitted=False,
        reserved_source_inference="not_run", independent_calibration="not_run",
        independent_confirmation="not_run", external_gain_established=False,
        deployment=False, stage5c_execution=False, smc_enabled=False)
    path = ROOT / cfg["reports"] / "analysis.json"
    if verify and not path.exists():
        raise ValueError("No completed analysis to replay")
    immutable_json(path, analysis)
    assert_current(identity)
    if verify:
        immutable_json(path.with_name("replay.json"), dict(analysis_sha256=sha256(path),
            models_replayed=len(reports), matching_sampling_streams=3,
            fixed_source_predictions_replayed=analysis["fixed_source_probe_predictions"],
            reserved_source_rows=0, result_source="cached_verified", all_checks_passed=True))
    beat(state="verified" if verify else "training_endpoints_verified", analysis_sha256=sha256(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit-only", "resume", "verify"):
        parser.add_argument("--" + name, action="store_true")
    parser.add_argument("--trial")
    parser.add_argument("--stop-at", type=int)
    args = parser.parse_args()
    if ((args.audit_only and args.verify) or (args.stop_at is not None and
            (not args.trial or args.audit_only or args.verify)) or (args.verify and args.trial)):
        parser.error("One explicit phase; a timing pilot must name its registered trial")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cfg, data, fold, identity = load()
    matrix = [(family, seed) for family in cfg["families"] for seed in cfg["seeds"]]
    if args.trial and args.trial not in {f"{family}_seed{seed}" for family, seed in matrix}:
        raise ValueError("Unregistered trial")
    root = ROOT / cfg["output"]
    root.mkdir(parents=True, exist_ok=True)
    if subprocess.run(["git", "check-ignore", "--quiet", str(root / "checkpoint.pt")], cwd=ROOT).returncode:
        raise ValueError("Training output must remain Git ignored")
    def beat(**values):
        event = dict(pid=os.getpid(), timestamp_unix=time.time(), **values)
        json_write(root / "heartbeat.json", event)
        with (root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("Resume atomic checkpoint")))
    with (root / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        immutable_json(root / "identity.json", identity)
        if args.audit_only:
            beat(state="preflight_pass", source_rows=len(data["sites"]), source_sites=cfg["sites"],
                 source_recordings=len(identity["training_recordings"]), registered_models=len(matrix),
                 source_bindings=len(identity["source_bindings"]), reserved_source_rows=0)
            return
        if args.verify:
            verify_all(cfg, data, fold, identity, beat, True)
            return
        for family, seed in matrix:
            key = f"{family}_seed{seed}"
            if args.trial and args.trial != key:
                continue
            folder = root / "trials" / key
            ti = trial_identity(identity, family, seed)
            if (folder / "complete.json").exists():
                completed(folder / "complete.json", ti, cfg["training"])
                beat(state="cached_verified_complete", trial=key)
                continue
            torch.manual_seed(seed)
            model = build_forecaster(identity["architectures"][family])
            beat(state="training_started", trial=key, source_rows=len(data["sites"]),
                 torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                 data_loader_workers=0, runtime="torch_cpu_arm64")
            fit = fit_trial(model, data, fold, seed=seed, settings=cfg["training"], identity=ti,
                directory=folder, resume=args.resume, stop_at=args.stop_at,
                heartbeat=lambda **values: beat(trial=key, **values))
            if not fit["complete"]:
                beat(state="timing_pilot_complete_not_full", trial=key, fit=fit)
                return
            assert_current(identity)
            immutable_json(folder / "complete.json", dict(identity=ti, fit=fit,
                checkpoint=str((folder / "checkpoint.pt").relative_to(ROOT)),
                checkpoint_sha256=sha256(folder / "checkpoint.pt")))
            beat(state="trial_complete", trial=key, fit_seconds=fit["seconds"])
        if not args.trial:
            verify_all(cfg, data, fold, identity, beat, False)
        else:
            beat(state="requested_trial_complete_all_models_not_yet_verified")


if __name__ == "__main__":
    main()
