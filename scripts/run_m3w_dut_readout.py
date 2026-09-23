"""One fixed DUT descriptive calibration-domain readout, never confirmation."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == "Darwin" and platform.machine() != "arm64":
    raise RuntimeError("Native arm64 environment required")
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key, "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.run_m3w_native_forecast import array_hash, immutable_json, json_write
from scripts.freeze_m3w_external_policy_chain import input_hash
from src.evaluation.m3w_recording_lineage import sha256
from src.evaluation.m3w_intake_admission import validate_admission
from src.evaluation.m3w_source_reservations import SourceReservations
from src.data_unification.m3w_causal_recordings import RecordingWindows
from src.world_model.m3w_frozen_policy_chain import FrozenView, ARMS
from src.evaluation.m3w_dut_readout import PrefixRecording, empty_statistics, accumulate, fixed_baselines, batch_infer

CONFIG = "configs/m3w_dut_readout_v1.json"
CODE = ("scripts/run_m3w_dut_readout.py", "src/evaluation/m3w_dut_readout.py",
        "tests/test_m3w_dut_readout.py", "src/evaluation/m3w_intake_admission.py",
        "src/evaluation/m3w_source_reservations.py", "src/data_unification/m3w_causal_recordings.py")


def specification():
    cfg = json.loads((ROOT/CONFIG).read_text())
    if (cfg["role"] != "calibration" or cfg["query_stride"] != 1 or cfg["annotation_stride"] != 1
            or cfg["history_steps"] != 8 or cfg["future_steps"] != 12
            or cfg["excluded_recordings"] != ["dut_intersection_04"]
            or any(cfg[k] for k in ("threshold_search", "model_selection", "training",
                "independent_confirmation", "deployment", "stage5c_executed", "smc_enabled"))):
        raise ValueError("Fixed descriptive calibration role and full query population required")
    bindings = {}
    def bind(path, expected=None):
        path = str(path)
        full = (ROOT/path).resolve()
        if not full.is_relative_to(ROOT):
            raise ValueError("Escaping readout dependency")
        digest = sha256(full)
        if expected is not None and digest != expected:
            raise ValueError("Changed readout dependency: "+path)
        if path in bindings and bindings[path] != digest:
            raise ValueError("Conflicting dependency")
        bindings[path] = digest
        return dict(path=path, sha256=digest)
    chain_ref = bind(cfg["policy_manifest"], cfg["policy_manifest_sha256"])
    chain = json.loads((ROOT/chain_ref["path"]).read_text())
    for path, digest in chain["source_bindings"].items():
        bind(path, digest)
    # Establish the fitted chain's exclusion, not a universal historical-use claim.
    predictors = json.loads((ROOT/chain["config"]["predictors"]["path"]).read_text())
    identity = predictors["identity"]
    if (set(predictors["source_sites"]) != {"coupa", "deathCircle", "gates", "hyang"}
            or "DUT" not in identity["excluded_from_all_fitting"]
            or len(identity["training_recordings"]) != 33
            or not all(r.split("/")[0] in predictors["source_sites"] for r in identity["training_recordings"])):
        raise ValueError("Producer lineage does not establish DUT exclusion")
    screen_ref = bind(cfg["intake_screen"])
    screen = json.loads((ROOT/cfg["intake_screen"]).read_text())
    use_evidence = [bind(cfg["source_use_disposition"]), bind(cfg["authorization"])]
    registration = str(Path(cfg["reports"])/"registration.md")
    use_evidence.append(bind(registration))
    reservation = SourceReservations(ROOT)
    records = {}
    for name, entry in sorted(screen["recordings"].items()):
        if name in cfg["excluded_recordings"]:
            continue
        if reservation.records[name]["reserved_role"] != "calibration":
            raise ValueError("DUT role changed")
        record = {k: entry[k] for k in ("cache_path", "metadata_sha256", "physical_scene")}
        metadata_ref = bind(str(Path(record["cache_path"])/"metadata.json"), record["metadata_sha256"])
        meta = json.loads((ROOT/metadata_ref["path"]).read_text())
        record.update(intake_screen=screen_ref, source_use_decision=dict(
            status="reviewed_for_declared_roles", reviewed_by="research_agent_under_explicit_delegation",
            decision_reference=cfg["source_use_disposition"], allowed_roles=["calibration"], evidence=use_evidence))
        for path, digest in validate_admission(ROOT, name, record, meta, "calibration").items():
            bind(path, digest)
        if meta["schema"]["future_inputs"] or meta["schema"]["central_velocity_used"] or meta["goals_constructed"]:
            raise ValueError("Unsupported causal source schema")
        for filename, detail in meta["artifacts"].items():
            bind(str(Path(record["cache_path"])/filename), detail["sha256"])
        records[name] = record
    if len(records) != 27 or {r["physical_scene"] for r in records.values()} != set(cfg["physical_sites"]):
        raise ValueError("Incomplete two-site DUT calibration inventory")
    # Raw hashes exclude exact renamed source copies; this does not prove geographic independence.
    external_hashes = {h for r in screen["recordings"].values() for h in r["source_file_hashes"].values()}
    fit_hashes = {h for p, h in chain["source_bindings"].items() if p.startswith("data/stage21_sdd")}
    if external_hashes & fit_hashes:
        raise ValueError("Byte-identical source overlap")
    for path in (CONFIG, *CODE):
        bind(path)
    manifest = dict(config=cfg, records=records, chain=chain_ref, source_bindings=bindings,
        observation_mode=cfg["observation_mode"], source_use="scoped_local_noncommercial_no_redistribution",
        historical_use="current_bound_fit_chain_excludes_DUT_universal_history_unknown",
        source_geography="author_identified_two_Dalian_sites_vs_Stanford_source_not_IID_certificate",
        calibration_domain_readout_authorized=True, independent_confirmation_authorized=False,
        risk_certificate=False, selected_policy=None, runtime=dict(python=platform.python_version(),
            architecture=platform.machine(), torch=torch.__version__, numpy=np.__version__),
        deployment=False)
    return cfg, manifest, chain


def query_receipt(scene, decision, labels, mask):
    return dict(frame=scene.frame_id, input_sha256=input_hash(scene),
        output_sha256=array_hash(decision["baseline"], decision["candidate"], decision["costs"],
                                *[decision["choices"][k] for k in ARMS]),
        label_sha256=array_hash(labels, mask))


def execute(cfg, manifest, chain, *, verify=False):
    public, directory = ROOT/cfg["reports"], ROOT/cfg["output"]
    manifest_hash = sha256(public/"manifest.json")
    started = time.monotonic()
    def beat(**kw):
        event = dict(pid=os.getpid(), elapsed_seconds=time.monotonic()-started, **kw)
        json_write(directory/"heartbeat.json", event)
        print(json.dumps(event), flush=True)
    views = {}
    for p in chain["predictors"]:
        view = f"{p['family']}_seed{p['seed']}"
        for h in (h for h in chain["cost_heads"] if h["view"] == view):
            key = view+"_"+h["head"]
            views[key] = FrozenView(ROOT, p, h, chain["architectures"][p["family"]], p["seed"])
    report_rows, replay_checks = [], []
    for name, record in manifest["records"].items():
        reader = RecordingWindows(ROOT/record["cache_path"])
        source = PrefixRecording(reader.points, reader.frame_order, name)
        done_path = directory/"records"/(name+".json")
        progress_path = directory/"records"/(name+"_partial.json")
        if done_path.exists() and not verify:
            done = json.loads(done_path.read_text())
            if done["manifest_sha256"] != manifest_hash or done["queries"] != len(source.queries):
                raise ValueError("Changed completed recording identity")
            report_rows.append(dict(recording=name, path=str(done_path.relative_to(ROOT)), sha256=sha256(done_path)))
            beat(state="cached_verified_record", recording=name)
            continue
        state = dict(manifest_sha256=manifest_hash, recording=name, site=record["physical_scene"],
            queries=len(source.queries), next_query=0, stats={v:empty_statistics() for v in views},
            probes={}, rolling_sha256="0"*64)
        if progress_path.exists() and not verify:
            state = json.loads(progress_path.read_text())
            if state["manifest_sha256"] != manifest_hash or state["queries"] != len(source.queries):
                raise ValueError("Resume identity changed")
        # Full execution; verification reproduces the three registered chunks per recording.
        step = cfg["checkpoint_queries"]
        probe_chunks = {0, (len(source.queries)//2//step)*step,
                        ((len(source.queries)-1)//step)*step}
        starts = sorted(probe_chunks) if verify else range(state["next_query"], len(source.queries), step)
        expected = json.loads(done_path.read_text()) if verify else None
        for begin in starts:
            frames = source.queries[begin:begin+step]
            scenes = [source.inputs(int(f)) for f in frames]
            predictions = {v:batch_infer(model, scenes) for v, model in views.items()}
            # Input construction and every view's decisions precede label access.
            receipts = {v:[] for v in views}
            for j, scene in enumerate(scenes):
                labels, mask = source.labels(scene.frame_id, scene.target_ids)
                controls = fixed_baselines(source, scene)
                for v in views:
                    out = predictions[v][j]
                    receipts[v].append(query_receipt(scene, out, labels, mask))
                    if not verify:
                        accumulate(state["stats"][v], scene, out, labels, mask, reader.metadata["agent_table"],
                            easy_cut=cfg["easy_normalized_ADE_max"], hard_cut=cfg["hard_normalized_ADE_min"],
                            controls=controls if v == next(iter(views)) else None)
            digest = hashlib.sha256(json.dumps(receipts, sort_keys=True).encode()).hexdigest()
            if verify:
                if digest != expected["probes"][str(begin)]:
                    raise ValueError("Frozen replay differs: "+name)
                replay_checks.append(dict(recording=name, begin=begin, queries=len(frames), exact=True))
            else:
                state["rolling_sha256"] = hashlib.sha256((state["rolling_sha256"]+digest).encode()).hexdigest()
                if begin in probe_chunks:
                    state["probes"][str(begin)] = digest
                state["next_query"] = begin+len(frames)
                json_write(progress_path, state)
            beat(state="replay_chunk" if verify else "evaluating", recording=name,
                 completed=begin+len(frames), total=len(source.queries), model_views=len(views))
        if not verify:
            immutable_json(done_path, state)
            report_rows.append(dict(recording=name, path=str(done_path.relative_to(ROOT)), sha256=sha256(done_path)))
    if verify:
        result = dict(result_source="cached_verified_fresh_fixed_chunk_replay", manifest_sha256=manifest_hash,
            all_checks_passed=True, scope="first_middle_last_128_query_chunks_not_second_full_readout",
            checks=replay_checks, elapsed_seconds=time.monotonic()-started)
        json_write(public/"verification.json", result)
    else:
        result = dict(result_source="fresh_run", manifest_sha256=manifest_hash, records=report_rows,
            views=list(views), dataset="DUT", physical_sites=2, recording_count=27,
            training=False, model_selection=False, calibration_domain_diagnostic=True,
            independent_confirmation=False, risk_certified=False, deployment=False,
            stage5c_executed=False, smc_enabled=False)
        immutable_json(public/"analysis.json", result)
    beat(state="complete", verification=verify)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if sum((args.freeze, args.run, args.verify)) != 1:
        parser.error("Choose one operation")
    cfg, manifest, chain = specification()
    public, directory = ROOT/cfg["reports"], ROOT/cfg["output"]
    if args.freeze:
        immutable_json(public/"manifest.json", manifest)
        print(json.dumps(dict(status="frozen_before_label_readout", recordings=27, sites=2,
                              bindings=len(manifest["source_bindings"]))))
        return
    if json.loads((public/"manifest.json").read_text()) != manifest:
        raise ValueError("Frozen readout identity changed")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/"runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(cfg["torch_threads"])
        torch.set_num_interop_threads(cfg["interop_threads"])
        execute(cfg, manifest, chain, verify=args.verify)


if __name__ == "__main__":
    main()
