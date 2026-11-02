"""Complete SOURCE-only fitting and frozen model-selection policy comparison."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time

if platform.system() == "Darwin" and platform.machine() != "arm64":
    raise RuntimeError("Native arm64 .venv-pytorch required before Torch")
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key, "4")
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_selection_forecast as frozen
from src.world_model import m3w_selection_controller as api
from src.world_model import m3w_frame_action_allocation as allocator
from src.world_model.m3w_european_source_forecast import baseline_numpy
from src.world_model.m3w_detector_forecaster import DetectorGeometry, pack_geometry
from src.evaluation import m3w_selection_forecast_readout as metrics
from src.evaluation.m3w_european_squares_roles import require_source_training

NAME = "european_selection_controller_v1"
PUBLIC = ROOT/"outputs/publication_readiness_2026_09"/NAME
PRIVATE = ROOT/"data/stage_cvpr2027_experiments"/NAME
CONFIG = ROOT/"configs/m3w_selection_controller_v1.json"; REG = PUBLIC/"registration_v2.json"
SOURCE = frozen.PACKED
CONTROLLER_STORE = ROOT/"outputs/publication_readiness_2026_09/european_detector_history_store_v1/complete.json"
SOURCE_STORE = frozen.STORE/"complete.json"
OUTCOMES = ROOT/"outputs/publication_readiness_2026_09/european_selection_forecast_readout_v1"
sha, write, once, artifact, checked = frozen.sha, frozen.write, frozen.once, frozen.artifact, frozen.checked
FILES = (Path(__file__), CONFIG, ROOT/"src/world_model/m3w_selection_controller.py",
         ROOT/"tests/test_m3w_selection_controller.py", PUBLIC/"protocol.md")
DEPENDENCIES = ("src/world_model/m3w_floor_relative.py", "src/world_model/m3w_european_source_intervention.py",
    "src/world_model/m3w_native_gain_harm.py", "src/world_model/m3w_multiaction_cost.py",
    "src/world_model/m3w_frame_action_allocation.py", "src/world_model/m3w_reference_cost_refit.py",
    "src/world_model/m3w_causal_descriptor_head.py", "src/evaluation/m3w_selection_forecast_readout.py",
    "src/evaluation/m3w_native_metrics.py")


def admission(register=False):
    cfg = json.loads(CONFIG.read_text()); _, roles, _, _, chain = frozen.admission()
    if (cfg["arms"] != list(api.ARMS) or cfg["fits"] != 18 or cfg["seeds"] != [17, 29, 43]
            or cfg["positive_harm_budget"] != .02 or cfg["easy_net_degradation_budget"] != .02
            or cfg["source_whitelist_rows"] != 149510 or cfg["selection_rows"] != 38102
            or cfg["training"]["query_batch_size"] != 32 or cfg["training"]["largest_source_cycles"] != 4
            or any(cfg[k] for k in ("model_selection_labels_for_training", "threshold_search", "calibration_access",
                "confirmation_access", "new_trajectory_training", "stage5c_executed", "smc_enabled"))):
        raise ValueError("Original complete SOURCE/selection and risk scope required")
    prior = json.loads((OUTCOMES/"verification.json").read_text())
    if not prior["all54_metric_groups_exact"] or prior["summary_sha256"] != sha(OUTCOMES/"summary.json"):
        raise ValueError("Previous exposed selection readout identity changed")
    store = json.loads(CONTROLLER_STORE.read_text()); ids = np.load(checked(store["arrays"]["ids"]), mmap_mode="r", allow_pickle=False)
    if len(ids) != 149510 or (np.diff(ids) <= 0).any(): raise ValueError("Original sorted SOURCE whitelist required")
    packed = json.loads((SOURCE/"receipt.json").read_text())
    for key in ("sites", "recordings", "frames", "agents", "history", "origin", "target_eval", "valid"):
        if sha(SOURCE/(key+".npy")) != packed["arrays"][key]: raise ValueError("Original SOURCE array changed")
    sites = np.load(SOURCE/"sites.npy", mmap_mode="r", allow_pickle=False)
    recordings = np.load(SOURCE/"recordings.npy", mmap_mode="r", allow_pickle=False)
    for rec in np.unique(recordings[ids]): require_source_training(roles, packed["records"][int(rec)])
    producers = [r for r in chain["lineage"] if r["representation"] == "history"]
    fit_scopes = []
    for row in producers:
        selected = ids[~np.isin(sites[ids], row["fit_sites"])]
        names = sorted(set(sites[selected]))
        if len(names) != 8 or set(names) & set(row["fit_sites"]): raise ValueError("Eight producer-excluded fitting sites required")
        fit_scopes.append(dict(trial=row["trial"], producer_fit_sites=row["fit_sites"], controller_fit_sites=names,
                              controller_fit_rows=len(selected), controller_fit_ids_sha256=frozen.fingerprint(selected)))
    value = dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        original_registration=artifact(PUBLIC/"registration.json"),
        pre_formal_training_numerical_repair="replace_parallel_scatter_gather_reductions_with_fixed_membership_products_both_arms_objective_unchanged",
        original_failed_recovery_pilot=artifact(PUBLIC/"pilot_failure_v1.json"),
        unchanged_verified_source_bank_freeze=artifact(PUBLIC/"source_bank_freeze.json"),
        dependencies={p:sha(ROOT/p) for p in DEPENDENCIES}, parent_forecast_registration=artifact(frozen.REG),
        selection_feature_freeze=artifact(frozen.PUBLIC/"feature_freeze.json"),
        selection_prediction_freeze=artifact(frozen.PUBLIC/"prediction_freeze.json"),
        previous_exposed_selection_readout=artifact(OUTCOMES/"complete.json"),
        previous_exposed_selection_verification=artifact(OUTCOMES/"verification.json"),
        source_packed_receipt=artifact(SOURCE/"receipt.json"), source_detector_store=artifact(SOURCE_STORE),
        source_whitelist=store["arrays"]["ids"], roles_sha256=sha(frozen.inputs.ROLES), fit_scopes=fit_scopes,
        fits=18, primary=cfg["primary"], selection_outcomes_already_exposed=True,
        model_selection_labels_for_training=False, calibration_access=False, confirmation_access=False,
        new_trajectory_training=False, stage5c_executed=False, smc_enabled=False)
    if register: once(REG, value)
    if json.loads(REG.read_text()) != value: raise ValueError("Registered controller definition changed")
    if not register:
        for p in (*FILES, REG):
            blob = subprocess.run(["git", "show", "HEAD:"+str(p.relative_to(ROOT))], cwd=ROOT, capture_output=True, check=True).stdout
            if hashlib.sha256(blob).hexdigest() != sha(p): raise ValueError("Commit controller contract before execution")
    return cfg, producers, ids


def runtime(cfg, phase):
    torch.set_num_threads(4); torch.set_num_interop_threads(1); start = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("Resume unchanged atomic checkpoints")))
    def guard():
        if shutil.disk_usage(ROOT).free < cfg["disk_reserve_bytes"]: raise OSError("Preserve10GiB reserve")
        if time.monotonic()-start > cfg["hard_runtime_limit_seconds"]: raise TimeoutError("12h phase boundary; resume unchanged")
        if sum(p.stat().st_size for p in PRIVATE.rglob("*") if p.is_file()) > cfg["private_cache_cap_bytes"]:
            raise OSError("Keep the registered private cache below1GiB")
    def beat(**kw):
        guard(); value = dict(pid=os.getpid(), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), phase=phase, **kw)
        write(PRIVATE/"heartbeat.json", value); print(json.dumps(value), flush=True)
    def receipt():
        return dict(pid=os.getpid(), seconds=time.monotonic()-start,
                    peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, registration_sha256=sha(REG))
    beat(state="runtime_ready", architecture=platform.machine(), device="cpu", threads=4, interop=1, workers=0)
    return beat, guard, receipt


def source_inputs(chain):
    packed = json.loads((SOURCE/"receipt.json").read_text())
    out = {k:np.load(SOURCE/(k+".npy"), mmap_mode="r", allow_pickle=False)
           for k in ("sites", "recordings", "frames", "agents", "history", "origin")}
    out["geometry"] = np.load(checked(chain["source_partial_geometry"]), mmap_mode="r", allow_pickle=False)
    ref = json.loads(SOURCE_STORE.read_text())["arrays"]["features"]
    out["detector_history"] = np.load(checked(ref), mmap_mode="r", allow_pickle=False)
    if len(out["sites"]) != packed["rows"]: raise ValueError("SOURCE packed row alignment changed")
    return out


def source_banks(resume=False):
    cfg, producers, whitelist = admission(); beat, guard, receipt = runtime(cfg, "source_banks")
    if (PUBLIC/"source_bank_freeze.json").exists(): raise ValueError("Complete SOURCE bank immutable")
    chain = json.loads(frozen.REG.read_text()); data = source_inputs(chain); refs = []
    for row in producers:
        trial = row["trial"]; ids = np.asarray(whitelist[~np.isin(data["sites"][whitelist], row["fit_sites"])])
        home = PRIVATE/"source_banks"/trial; home.mkdir(parents=True, exist_ok=True); finish = home/"complete.json"
        binding = dict(registration_sha256=sha(REG), checkpoint=row["checkpoint"],
                       ids_sha256=frozen.fingerprint(ids), rows=len(ids), batch_size=128)
        if finish.exists():
            saved = json.loads(finish.read_text())
            if not resume or saved["binding"] != binding: raise ValueError("Explicit unchanged bank resume required")
            checked(saved["prediction"]); refs.append(artifact(finish)); continue
        path, progress = home/"prediction.npy", home/"progress.json"
        if path.exists():
            if not resume: raise ValueError("Explicit partial SOURCE bank resume required")
            saved = json.loads(progress.read_text())
            if saved["binding"] != binding: raise ValueError("SOURCE inference binding changed")
            pred = np.load(path, mmap_mode="r+", allow_pickle=False); start = saved["rows_complete"]
            if frozen.fingerprint(pred[:start]) != saved["prefix_sha256"]: raise ValueError("Committed prefix changed")
        else:
            pred = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(ids),12,2)); start = 0
            pred[:] = 0; pred.flush(); write(progress, dict(binding=binding, rows_complete=0, prefix_sha256=frozen.fingerprint(pred[:0])))
        state = frozen.checked_state(row)
        model = frozen.producers.api.ActiveDetectorForecaster(row["baseline_index"], "history", width=64, heads=4, layers=2)
        model.load_state_dict(state["model"]); model.eval(); del state
        norm = row["normalizer"]
        view = DetectorGeometry(data["geometry"], data["detector_history"], np.asarray(norm["mean"]), np.asarray(norm["std"]), norm["clip"])
        for lo in range(start, len(ids), 128):
            guard(); hi = min(lo+128, len(ids))
            with torch.no_grad(): pred[lo:hi] = model(pack_geometry(view[ids[lo:hi]])).numpy()
            if hi % 1280 == 0 or hi == len(ids):
                pred.flush(); write(progress, dict(binding=binding, rows_complete=hi, prefix_sha256=frozen.fingerprint(pred[:hi])))
                beat(state="source_bank_inference", trial=trial, rows=hi, total=len(ids))
        previous = json.loads((frozen.producers.PUBLIC/"groups"/(trial+"_history.json")).read_text())
        if (frozen.producers.source.array_hash(ids) != previous["ids_sha256"]
                or frozen.producers.source.array_hash(pred) != previous["prediction_sha256"]["history"]):
            raise ValueError("Fresh SOURCE forecasts differ from exact original out-of-producer-fit bank")
        once(finish, dict(binding=binding, prediction=artifact(path), prior_active_producer_prediction_exact=True,
                          ids_sha256=frozen.fingerprint(ids), future_targets_used=False))
        refs.append(artifact(finish)); beat(state="source_bank_frozen", trial=trial, completed=len(refs), total=9)
    once(PUBLIC/"source_bank_freeze.json", dict(**receipt(), banks=refs, all9_old_prediction_hashes_exact=True,
        future_targets_used=False, new_optimizer_updates=0))


def fitting_data(row, source, whitelist):
    ids = np.asarray(whitelist[~np.isin(source["sites"][whitelist], row["fit_sites"])])
    ref = json.loads((PRIVATE/"source_banks"/row["trial"]/"complete.json").read_text())
    if ref["ids_sha256"] != frozen.fingerprint(ids): raise ValueError("SOURCE cost rows changed")
    neural = np.load(checked(ref["prediction"]), mmap_mode="r", allow_pickle=False)
    origin = source["origin"][ids]
    physical = np.stack([baseline_numpy(source["history"][ids], k)-origin[:,None] for k in range(6)], 1)
    rollouts = np.concatenate((physical, neural[:, None].astype(float)), 1)
    causal = api.features(geometry=source["geometry"][ids], detector_history=source["detector_history"][ids],
        rollouts=rollouts, floor_index=row["baseline_index"], recordings=source["recordings"][ids],
        frames=source["frames"][ids], ids=ids, sites=source["sites"][ids])
    # Only the producer-excluded SOURCE IDs enter the supervised accessor.
    target = np.load(SOURCE/"target_eval.npy", mmap_mode="r", allow_pickle=False)[ids]-origin[:,None]
    valid = np.load(SOURCE/"valid.npy", mmap_mode="r", allow_pickle=False)[ids]
    y, _ = api.labels(rollouts, target, valid, row["baseline_index"], row["easy_cut"])
    return api.training_data(causal, y)


def fit_path(trial, arm, family="fits"):
    return PRIVATE/family/(trial+"_"+arm+".pt.gz")


def fit_one(data, row, arm, cfg, beat, guard, *, family="fits", resume=False, stop_at=None):
    return api.fit(data, arm=arm, seed=row["seed"], settings=cfg["training"],
        identity=dict(registration_sha256=sha(REG), source_bank=artifact(PRIVATE/"source_banks"/row["trial"]/"complete.json"),
                      producer=row["checkpoint"], source_packed_receipt=artifact(SOURCE/"receipt.json")),
        path=fit_path(row["trial"], arm, family), heartbeat=lambda **v:beat(trial=row["trial"], **v),
        guard=guard, resume=resume, stop_at=stop_at)


def assert_recovery(left, right):
    for k in left:
        if k != "seconds": api.exact(left[k], right[k])


def pilot():
    cfg, producers, whitelist = admission(); beat, guard, receipt = runtime(cfg, "pilot")
    if (PUBLIC/"pilot.json").exists(): raise ValueError("Completed controller pilot immutable")
    source = source_inputs(json.loads(frozen.REG.read_text())); row = producers[0]
    data = fitting_data(row, source, whitelist); cells = []
    for arm in api.ARMS:
        fit_one(data, row, arm, cfg, beat, guard, family="pilot_deterministic_v2", stop_at=50)
        recovered = fit_one(data, row, arm, cfg, beat, guard, family="pilot_deterministic_v2", stop_at=100, resume=True)
        direct = fit_one(data, row, arm, cfg, beat, guard, family="pilot_direct_deterministic_v2", stop_at=100)
        assert_recovery(recovered, direct)
        cells.append(dict(arm=arm, parameters=direct["parameters"], seconds=direct["seconds"],
                          updates=direct["step"], budget=direct["budget"], exact_scientific_recovery=True,
                          first_trace=direct["trace"][0], final_trace=direct["trace"][-1]))
    once(PUBLIC/"pilot.json", dict(**receipt(), engineering_pass=True, cells=cells,
        actual_optimizer_updates=400, source_cost_sites=sorted(set(data["causal"]["sites"])),
        source_cost_rows=len(data["causal"]["ids"]), selection_outcomes_scored=False,
        projected_primary_fit_seconds=sum(c["seconds"]/100*c["budget"] for c in cells)*9,
        projection_excludes_preparation_refits_and_policy_allocation=True))


def train(resume=False):
    cfg, producers, whitelist = admission(); beat, guard, receipt = runtime(cfg, "train")
    if (PUBLIC/"training_freeze.json").exists(): raise ValueError("Complete cost training immutable")
    if not json.loads((PUBLIC/"pilot.json").read_text())["engineering_pass"]: raise ValueError("Actual training/recovery pilot required")
    source = source_inputs(json.loads(frozen.REG.read_text())); refs = []; updates = 0; refit_updates = 0
    for row in producers:
        data = fitting_data(row, source, whitelist); states = []
        for arm in api.ARMS:
            state = fit_one(data, row, arm, cfg, beat, guard, resume=resume and fit_path(row["trial"], arm).exists())
            if state["step"] != state["budget"] or int(state["sampler"]["counts"].min()) < 4:
                raise ValueError("Complete budget and every-query four visits required")
            if row == producers[0]:
                other = fit_one(data, row, arm, cfg, beat, guard, family="full_refit",
                    resume=resume and fit_path(row["trial"], arm, "full_refit").exists())
                assert_recovery(state, other); refit_updates += state["step"]
                once(PUBLIC/("first_full_refit_"+arm+".json"), dict(scientific_state_exact=True, updates=state["step"],
                     checkpoint=artifact(fit_path(row["trial"], arm)), refit=artifact(fit_path(row["trial"], arm, "full_refit"))))
            path = PUBLIC/"fits"/(row["trial"]+"_"+arm+".json")
            once(path, dict(trial=row["trial"], seed=row["seed"], fold=row["fold"], arm=arm,
                checkpoint=artifact(fit_path(row["trial"], arm)), parameters=state["parameters"],
                step=state["step"], budget=state["budget"], seconds=state["seconds"], source_sites=state["source_sites"],
                producer_sites=row["fit_sites"], fitting_rows=len(state["fit_ids"]), fit_ids_sha256=frozen.fingerprint(state["fit_ids"]),
                minimum_query_visits=int(state["sampler"]["counts"].min()), never_drawn_queries=int((state["sampler"]["counts"]==0).sum()),
                draw_hash=state["draw_hash"], unknown_draws=state["unknown_draws"], unknown_supervised_rows=0,
                first_trace=state["trace"][0], final_trace=state["trace"][-1], registration_sha256=sha(REG)))
            refs.append(artifact(path)); updates += state["step"]; states.append(state)
            beat(state="controller_fit_complete", trial=row["trial"], arm=arm, fits=len(refs), total=18)
        for key in ("initial_model", "parameters", "budget", "sampler", "draw_hash", "fit_ids", "preprocess", "torch_rng"):
            api.exact(states[0][key], states[1][key])
        del data, states
    once(PUBLIC/"training_freeze.json", dict(**receipt(), fits=refs, primary_optimizer_updates=updates,
        exact_full_refit_updates=refit_updates, source_only=True, selection_labels_used=False, new_trajectory_training=False))


def selection_bank(row):
    data = frozen.load_features()
    forecast = json.loads((frozen.PUBLIC/"prediction_freeze.json").read_text())
    model = next(json.loads(checked(r).read_text()) for r in forecast["models"]
                 if Path(r["path"]).stem == row["trial"]+"_history")
    neural = np.load(checked(model["prediction"]), mmap_mode="r", allow_pickle=False)
    physical = np.load(checked(forecast["physical_baselines"]), mmap_mode="r", allow_pickle=False)
    paths = np.concatenate((physical, neural[:,None]), 1)
    relative = paths-data["origin"][:, None, None]
    causal = api.features(geometry=data["geometry"], detector_history=data["detector_history"], rollouts=relative,
        floor_index=row["baseline_index"], recordings=data["recordings"], frames=data["frames"], ids=data["ids"], sites=data["sites"])
    return data, paths, causal


def fitted():
    freeze = json.loads((PUBLIC/"training_freeze.json").read_text()); out = {}
    for ref in freeze["fits"]:
        row = json.loads(checked(ref).read_text()); checked(row["checkpoint"])
        if row["step"] != row["budget"] or row["minimum_query_visits"] < 4 or set(row["source_sites"]) & set(row["producer_sites"]):
            raise ValueError("Complete source-only producer-excluded cost endpoints required")
        out[row["trial"], row["arm"]] = row
    if len(out) != 18: raise ValueError("All18cost endpoints required")
    return out


def infer(resume=False):
    cfg, producers, _ = admission(); beat, guard, receipt = runtime(cfg, "infer")
    if (PUBLIC/"policy_freeze.json").exists(): raise ValueError("Complete policy freeze immutable")
    models = fitted(); refs = []
    for row in producers:
        data, _, causal = selection_bank(row); trial = row["trial"]
        for arm in api.ARMS:
            result = PUBLIC/"policies"/(trial+"_"+arm+".json")
            if result.exists():
                saved = json.loads(result.read_text())
                if not resume or saved["registration_sha256"] != sha(REG): raise ValueError("Explicit unchanged policy resume required")
                checked(saved["prediction"]); checked(saved["choices"]); refs.append(artifact(result)); continue
            state = api.read_checkpoint(checked(models[trial,arm]["checkpoint"])); pred = api.predict(state, causal)
            chosen = {}; allocation = {}
            for suffix, allowed in (("", causal["allowed"]), *(([("_physical", causal["allowed"].copy())]) if arm=="context" else [])):
                if suffix: allowed[:,6] = False
                progress = PRIVATE/"allocation_progress"/(trial+"_"+arm+suffix+".gz")
                values, _, info = allocator.allocate(pred, allowed, data["recordings"], data["frames"], data["ids"],
                    seconds=cfg["solver_seconds_per_query"], checkpoint=lambda v:api.save_checkpoint(progress, v),
                    heartbeat=lambda **v:beat(state="selection_causal_allocation", trial=trial, arm=arm+suffix, **v),
                    resume_state=api.read_checkpoint(progress) if resume and progress.exists() else None)
                chosen.update({k+suffix:v for k,v in values.items()}); allocation[suffix or "full"] = info
            home = PRIVATE/"policies"/(trial+"_"+arm); home.mkdir(parents=True, exist_ok=True)
            frozen.inputs.store_array(home/"prediction.npy", pred)
            keys = list(chosen); frozen.inputs.store_array(home/"choices.npy", np.stack([chosen[k] for k in keys]))
            once(result, dict(trial=trial, arm=arm, prediction=artifact(home/"prediction.npy"),
                choices=artifact(home/"choices.npy"), policy_names=keys, allocation=allocation,
                checkpoint=models[trial,arm]["checkpoint"], inference_features=frozen.fingerprint(causal["x"]),
                registration_sha256=sha(REG), future_labels_read=False, selection_rows=len(data["ids"])))
            refs.append(artifact(result)); beat(state="causal_policy_frozen", trial=trial, arm=arm, count=len(refs), total=18)
    once(PUBLIC/"policy_freeze.json", dict(**receipt(), policies=refs, training_freeze=artifact(PUBLIC/"training_freeze.json"),
        labels_read_for_this_comparison=False, threshold_search=False, calibration_access=False, confirmation_access=False))


def selection_labels(data):
    complete = json.loads((OUTCOMES/"complete.json").read_text()); yy, mm = [], []
    for ref in complete["label_receipts"]:
        label = json.loads(checked(ref).read_text()); yy.append(np.load(checked(label["target"]), allow_pickle=False))
        mm.append(np.load(checked(label["valid"]), allow_pickle=False))
    y, valid = np.concatenate(yy), np.concatenate(mm)
    if len(y) != len(data["ids"]) or len(y) != 38102: raise ValueError("All original selection labels required")
    return y, valid


def selected_paths(paths, choice, floor_index):
    return paths[np.arange(len(paths)), np.where(choice < 0, floor_index, choice)]


def summarize(rows, cfg):
    sites = sorted({r["site"] for r in rows})
    expected = {(f"single{f}_seed{s}", site) for f in range(3) for s in (17,29,43) for site in sites}
    if len(rows) != 54 or len(sites) != 6 or {(r["trial"],r["site"]) for r in rows} != expected:
        raise ValueError("All9producers and six selection sites required")
    ci = lambda fn: metrics.interval(rows, fn, expected_sites=sites, resamples=cfg["bootstrap_resamples"], seed=cfg["bootstrap_seed"])
    names = tuple(rows[0]["policies"])
    accuracy = {a:ci(lambda r,a=a:r["policies"][a]["known_ADE_gain_floor_percent"]) for a in names}
    effects = {k:ci(lambda r,k=k:r["primary_joint_minus_independent"][k]) for k in ("known_pp","completion_lower_pp","completion_upper_pp")}
    context_effect = {k:ci(lambda r,k=k:r["context_minus_row_joint_matched"][k]) for k in effects}
    neural_effect = {k:ci(lambda r,k=k:r["context_full_minus_physical_joint"][k]) for k in effects}
    safety = {}
    for name in names:
        views = [v for r in rows for v in r["policies"][name]["recordings"].values()]
        easy = [v["known_easy_net_degradation"] for v in views if v["known_easy_net_degradation"] is not None]
        safety[name] = dict(required_record_views=len(views), supported_record_views=sum(v["supported"] for v in views),
            known_all_risk_violations=sum(v["known_all_risk"] is not None and v["known_all_risk"] > .02 for v in views),
            known_easy_risk_violations=sum(v["known_easy_risk"] is not None and v["known_easy_risk"] > .02 for v in views),
            undefined_easy_views=sum(v["completion_easy_risk_upper"] is None for v in views),
            worst_known_easy_net_degradation_percent=100*max(easy) if easy else None)
    primary_pass = effects["completion_lower_pp"]["CI95"][0] > 0
    risk_pass = safety["context_joint_matched"]["supported_record_views"] == safety["context_joint_matched"]["required_record_views"]
    return dict(groups=54, selection_localities=6, unique_target_rows=38102, unique_recordings=28,
        accuracy=accuracy, safety=safety, primary=effects, context_effect=context_effect, neural_action_effect=neural_effect,
        primary_pass=primary_pass, original_record_risk_pass=risk_pass, policy_candidate_pass=primary_pass and risk_pass,
        selection_development_not_final_confirmation=True, selection_labels_used_for_training=False,
        calibration_access=False, confirmation_access=False, pair_interactions_optimized=False,
        new_trajectory_training=False, deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)


def evaluate(verify=False, resume=False):
    cfg, producers, _ = admission(); beat, guard, receipt = runtime(cfg, "verify" if verify else "eval")
    finish = PUBLIC/("verification.json" if verify else "complete.json")
    if finish.exists(): raise ValueError("Complete selection controller readout immutable")
    freeze = json.loads((PUBLIC/"policy_freeze.json").read_text())
    if freeze["training_freeze"] != artifact(PUBLIC/"training_freeze.json") or len(freeze["policies"]) != 18:
        raise ValueError("Freeze all18causal policies before selection outcomes")
    saved = {}
    for ref in freeze["policies"]:
        value = json.loads(checked(ref).read_text()); checked(value["prediction"]); checked(value["choices"])
        saved[value["trial"],value["arm"]] = value
    rows, refs = [], []
    for row in producers:
        data, paths, causal = selection_bank(row); target, valid = selection_labels(data); floor = paths[:,row["baseline_index"]]
        policies = dict(floor=floor, raw_history=paths[:,6]); trial = row["trial"]
        for arm in api.ARMS:
            ref = saved[trial,arm]; prediction = np.load(checked(ref["prediction"]), allow_pickle=False)
            if verify:
                fresh = api.predict(api.read_checkpoint(checked(ref["checkpoint"])), causal)
                np.testing.assert_array_equal(fresh, prediction)
            choices = np.load(checked(ref["choices"]), allow_pickle=False)
            for name, choice in zip(ref["policy_names"], choices):
                allowed = causal["allowed"].copy()
                if name.endswith("_physical"): allowed[:,6] = False
                for group in causal["groups"]:
                    if not allocator.feasible(prediction[group], allowed[group], choice[group]):
                        raise ValueError("Frozen policy violates its predicted causal constraints")
                policies[arm+"_"+name] = selected_paths(paths, choice, row["baseline_index"])
            for suffix in ("", "_physical") if arm=="context" else ("",):
                left = choices[ref["policy_names"].index("independent"+suffix)]
                right = choices[ref["policy_names"].index("joint_matched"+suffix)]
                if any((left[g]>=0).sum() != (right[g]>=0).sum() for g in causal["groups"]):
                    raise ValueError("Matched intervention counts must hold at each query")
        for site in sorted(set(data["sites"])):
            at = data["sites"] == site
            scored = {k:metrics.policy(v[at], floor[at], paths[at,1], target[at], valid[at], data["recordings"][at],
                easy_cut=row["easy_cut"], hard_cut=row["hard_cut"]) for k,v in policies.items()}
            prior = json.loads((OUTCOMES/"groups"/(trial+"_"+str(site)+".json")).read_text())
            if scored["raw_history"] != prior["policies"]["history"] or scored["floor"] != prior["policies"]["floor"]:
                raise ValueError("Original raw-neural and no-intervention controls changed")
            comparisons = dict(primary_joint_minus_independent=("context_independent", "context_joint_matched"),
                context_minus_row_joint_matched=("row_joint_matched", "context_joint_matched"),
                context_full_minus_physical_joint=("context_joint_matched_physical", "context_joint_matched"))
            pairs = {key:metrics.paired(policies[a][at], policies[b][at], floor[at], target[at], valid[at])
                     for key,(a,b) in comparisons.items()}
            independent = scored["context_independent"]["recordings"]
            matched_support = {r:dict(nonempty=independent[r]["selected"]>0,
                status="nonempty_utility_comparison_only" if independent[r]["selected"]>0 else "not_estimable_empty_count_matched_control")
                for r in independent}
            value = dict(trial=trial, seed=row["seed"], source_fold=row["fold"], site=str(site), policies=scored, **pairs,
                         count_matched_record_support=matched_support,
                         policy_freeze_sha256=sha(PUBLIC/"policy_freeze.json"), registration_sha256=sha(REG))
            path = PUBLIC/"groups"/(trial+"_"+str(site)+".json")
            if verify or path.exists() and resume:
                if value != json.loads(path.read_text()): raise ValueError("Complete controller metrics replay differs")
            else: once(path, value)
            rows.append(value); refs.append(artifact(path))
        beat(state="selection_controller_group_complete", trial=trial, groups=len(rows), total=54)
    summary = summarize(rows, cfg)
    if verify:
        if summary != json.loads((PUBLIC/"summary.json").read_text()): raise ValueError("Canonical summary replay differs")
        once(finish, dict(**receipt(), all18_cost_inference_exact=True, all54_frozen_action_metric_groups_exact=True,
            all_queries_predicted_feasibility_and_matched_counts_checked=True, optimization_reexecuted=False,
            complete_sha256=sha(PUBLIC/"complete.json"), summary_sha256=sha(PUBLIC/"summary.json")))
    else:
        once(PUBLIC/"summary.json", summary); once(finish, dict(**receipt(), groups=refs,
            policy_freeze=artifact(PUBLIC/"policy_freeze.json"), summary_sha256=sha(PUBLIC/"summary.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "banks", "pilot", "train", "infer", "eval", "verify"))
    parser.add_argument("--resume", action="store_true"); args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True); PUBLIC.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase == "register": admission(True)
        elif args.phase == "banks": source_banks(args.resume)
        elif args.phase == "pilot": pilot()
        elif args.phase == "train": train(args.resume)
        elif args.phase == "infer": infer(args.resume)
        else: evaluate(args.phase == "verify", args.resume)
