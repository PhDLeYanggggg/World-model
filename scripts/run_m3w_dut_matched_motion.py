"""Freeze and test matched EqMotion and motion forecasts on disjoint DUT queries."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_dut_readout as old
from src.data_unification.m3w_causal_recordings import BASELINES
from src.evaluation import m3w_dut_matched_motion as method

ROOT, np, torch = old.ROOT, old.np, old.torch
NAME = "dut_matched_motion_v1"
OUT = ROOT / "outputs/publication_readiness_2026_09" / NAME
PRIVATE = ROOT / "data/stage_cvpr2027_experiments" / NAME
SEEDS, BATCH_QUERIES = (17, 29, 43), 16
RAW_MOTION = (*BASELINES, "ols4")
DAMP, OLS = "damped_velocity_005", "ols4"


def artifact(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=old.sha256(path), bytes=path.stat().st_size)


def read(path):
    return json.loads(path.read_text())


def register():
    cfg, current, chain = old.specification()
    manifest = ROOT / cfg["reports"] / "manifest.json"
    prior = ROOT / cfg["reports"] / "analysis.json"
    recorded = read(prior)
    if (current != read(manifest) or recorded["manifest_sha256"] != old.sha256(manifest)
            or recorded["independent_confirmation"] or recorded["recording_count"] != 27):
        raise ValueError("Require exactly the verified, previously exposed DUT roster")
    f = OUT / "registration.json"
    old.immutable_json(f, dict(code=artifact(Path(__file__)), method=artifact(Path(method.__file__)),
        prior=artifact(prior), source=artifact(manifest), query_rule="first observed eligible time then skip to >= prior+20",
        input="same raw eight-step target/peer prefix; all past-eligible targets",
        primary="direct raw EqMotion vs previous strongest damped005 and OLS4 on complete futures and full grid",
        matched="frozen strict nominations, common nonzero analytic directions, equal per-row peak from CV",
        split="same two exposed physical sites; no new roles or primary protocol changes",
        controls=list(RAW_MOTION), easy_cut=cfg["easy_normalized_ADE_max"], hard_cut=cfg["hard_normalized_ADE_min"],
        inherited_easy_limit=cfg["easy_degradation_max"], bootstrap=dict(draws=3000, seed=917, unit="physical_site"),
        seeds=list(SEEDS), batch_queries=BATCH_QUERIES, torch_threads=4, workers=0,
        no_refit=True, no_target_tuning=True, original_2pct=True, confirmation_access=False))
    return old.sha256(f), cfg, current, chain


def models(chain):
    result = {}
    for p in chain["predictors"]:
        if p["family"] != "eqmotion":
            continue
        seed = p["seed"]
        head, = [h for h in chain["cost_heads"] if h["view"] == f"eqmotion_seed{seed}"
                 and h["head"] == "bounded_fraction"]
        result[seed] = old.FrozenView(ROOT, p, head, chain["architectures"]["eqmotion"], seed)
    if set(result) != set(SEEDS):
        raise ValueError("Incomplete fixed three-seed EqMotion chain")
    return result


def motion(record, scene):
    frame = scene.frame_id
    prefix = record.rows(frame - 7, frame + 1)
    h = []
    for agent in scene.target_ids:
        p = prefix[prefix[:, 1] == agent]
        p = p[np.argsort(p[:, 0])]
        np.testing.assert_array_equal(p[:, 0], np.arange(frame - 7, frame + 1))
        h.append(p[:, 2:])
    h = np.asarray(h).reshape(-1, 8, 2)
    a = dict(zip(BASELINES, old.fixed_baselines(record, scene), strict=True))
    a[OLS] = method.ols4(h)
    return a


def query_predictions(scene, decision, analytic):
    loc = np.searchsorted(scene.agent_ids, scene.target_ids)
    cv = decision["baseline"][loc]
    raw_cv = analytic["constant_velocity_causal_fd"]
    np.testing.assert_allclose(cv, raw_cv, atol=2e-5, rtol=1e-7)
    candidate = decision["candidate"][loc]
    active = decision["choices"]["strict"][loc]
    strict = np.where(active[:, None, None], candidate, cv)
    raw = np.where(decision["choices"]["uncontrolled"][loc, None, None], candidate, cv)
    # Form analytic deltas in raw precision to avoid treating CV conversion roundoff as a direction.
    controls = {k: cv + (analytic[k] - raw_cv) for k in (DAMP, OLS)}
    shape, common = method.matched(cv, candidate, active, controls)
    return dict(cv=cv, eqmotion_raw=raw, strict=strict, **analytic,
                **{f"matched_{k}": p for k, p in shape.items()}), int(active.sum()), int(common.sum())


def run_record(name, record, views, cfg, reg):
    reader = old.RecordingWindows(ROOT / record["cache_path"])
    source = old.PrefixRecording(reader.points, reader.frame_order, name)
    queries = method.disjoint_queries(source.queries)
    scenes, kinds = [], []
    forecasts, totals = {s: [] for s in SEEDS}, {s: np.zeros(2, dtype=int) for s in SEEDS}
    for j in range(0, len(queries), BATCH_QUERIES):
        part = [source.inputs(int(f)) for f in queries[j:j + BATCH_QUERIES]]
        # Preserve an empty query in the roster; it supplies no target rows.
        scenes.extend(part)
        kinds.extend(reader.metadata["agent_table"][str(a)]["agent_type"] for s in part for a in s.target_ids)
        analytic = [motion(source, s) for s in part]
        for seed in SEEDS:
            for s, d, a in zip(part, old.batch_infer(views[seed], part), analytic, strict=True):
                p, n, m = query_predictions(s, d, a)
                forecasts[seed].append(p)
                totals[seed] += [n, m]
    paths = {s: {k: np.concatenate([p[k] for p in f]) for k in f[0]} for s, f in forecasts.items()}
    prediction_hash = old.array_hash(queries, *[v for s in SEEDS for v in paths[s].values()])
    # All forecasts and matched intervention masks precede the first label read.
    labels, masks = zip(*(source.labels(s.frame_id, s.target_ids) for s in scenes))
    y, valid = np.concatenate(labels), np.concatenate(masks)
    scales = np.concatenate([s.scales for s in scenes])
    cut, hard = cfg["easy_normalized_ADE_max"], cfg["hard_normalized_ADE_min"]
    cells = {}
    for seed, p in paths.items():
        cv = p["cv"]
        cv_mass = float(np.linalg.norm(cv - y, axis=-1)[valid].sum() / 12)
        pairs = {f"raw_vs_{k}": method.paired_summary(p[k], p["eqmotion_raw"], y, valid, scales, cut, hard)
                 for k in ("cv", DAMP, OLS)}
        for k in (DAMP, OLS):
            pairs[f"strict_vs_{k}"] = method.paired_summary(p[k], p["strict"], y, valid, scales, cut, hard)
            pairs[f"matched_vs_{k}"] = method.paired_summary(p[f"matched_{k}"], p["matched_neural"],
                                                             y, valid, scales, cut, hard)
        cells[str(seed)] = dict(cv_mass=cv_mass, strict=int(totals[seed][0]), common=int(totals[seed][1]),
            arms={k: method.arms_summary(cv, v, y, valid, scales, cut, hard, np.asarray(kinds)) for k, v in p.items()},
            paired=pairs)
    return dict(registration=reg, site=record["physical_scene"], recording=name, queries=len(queries),
        original_queries=len(source.queries), targets=len(y), complete=int(valid.all(1).sum()),
        partial=int((valid.any(1) & ~valid.all(1)).sum()), absent=int((~valid.any(1)).sum()),
        prediction_sha256=prediction_hash, labels_sha256=old.array_hash(y, valid), by_seed=cells)


def summarize(records):
    grouped = {s: [r for r in records if r["site"] == s] for s in sorted({r["site"] for r in records})}
    if len(grouped) != 2 or len(records) != 27:
        raise ValueError("Require all 27 recordings and the two physical sites")
    site = {}
    for s, recs in grouped.items():
        site[s] = {}
        for seed in SEEDS:
            cells = [r["by_seed"][str(seed)] for r in recs]
            arms, pairs = {}, {}
            den = sum(c["cv_mass"] for c in cells)
            for k in cells[0]["arms"]:
                parts = [c["arms"][k] for c in cells]
                slices = {}
                for subset in parts[0]["slices"]:
                    z = [p["slices"][subset] for p in parts]
                    n = sum(p["n"] for p in z)
                    value = dict(n=n, fde=sum(p["fde_sum"] for p in z) / n if n else None)
                    if subset != "endpoint":
                        mass, ref = (sum(p[f] for p in z) for f in ("ade_sum", "cv_sum"))
                        value.update(ade=mass / n if n else None, delta_pct=100. * (mass - ref) / ref if ref else None,
                            worst_record_p95=max((p["p95"] for p in z if p["p95"] is not None), default=None))
                    slices[subset] = value
                arms[k] = dict(slices=slices, edits=sum(v["edits"] for v in parts),
                    peak_sum=sum(v["peak_sum"] for v in parts), lower_pp=100. * sum(v["lower_gain"] for v in parts) / den,
                    undefined_selected_easy=sum(v["slices"]["selected_easy"]["cv_sum"] <= 0 for v in parts))
            for k in cells[0]["paired"]:
                z = [c["paired"][k] for c in cells]
                comp = sum(v["complete_ref"] for v in z)
                pairs[k] = dict(lower_pp=100. * sum(v["lower"] for v in z) / den,
                                upper_pp=100. * sum(v["upper"] for v in z) / den,
                                complete_gain_pct=100. * sum(v["complete_gain"] for v in z) / comp)
            site[s][str(seed)] = dict(arms=arms, paired=pairs, strict=sum(c["strict"] for c in cells),
                                     common=sum(c["common"] for c in cells))
    flat = [v for group in site.values() for v in group.values()]
    d = np.random.default_rng(917).integers(2, size=(3000, 2))
    pairs = {}
    for k in flat[0]["paired"]:
        pairs[k] = {}
        for f in flat[0]["paired"][k]:
            x = np.array([np.mean([v["paired"][k][f] for v in group.values()]) for group in site.values()])
            pairs[k][f] = dict(point=float(x.mean()), sites=x.tolist(),
                               interval95=np.quantile(x[d].mean(1), [.025, .975]).tolist())
    arms = {}
    for k in flat[0]["arms"]:
        p = [v["arms"][k] for v in flat]
        arms[k] = dict(
            complete_ade=float(np.mean([v["slices"]["complete"]["ade"] for v in p])),
            endpoint_fde=float(np.mean([v["slices"]["endpoint"]["fde"] for v in p])),
            lower_pp=float(np.mean([v["lower_pp"] for v in p])),
            worst_easy_pct=max((v["slices"]["easy"]["delta_pct"] for v in p
                                if v["slices"]["easy"]["delta_pct"] is not None), default=None),
            edits=sum(v["edits"] for v in p), undefined_selected_easy=sum(v["undefined_selected_easy"] for v in p))
    return dict(site=site, arms=arms, paired=pairs, strict=sum(v["strict"] for v in flat),
                common=sum(v["common"] for v in flat))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reg, cfg, manifest, chain = register()
        views = models(chain)
        output = []
        for name, rec in manifest["records"].items():
            f = PRIVATE / f"{name}.json"
            if args.replay and not f.exists():
                raise ValueError("Replay needs every prior record checkpoint")
            if args.replay or not f.exists():
                old.immutable_json(f, run_record(name, rec, views, cfg, reg))
            row = read(f)
            if row["registration"] != reg or row["recording"] != name:
                raise ValueError("Changed checkpoint identity")
            output.append(row)
            beat = dict(pid=os.getpid(), seconds=round(time.monotonic() - start, 3), recording=name,
                        complete=len(output), queries=row["queries"], targets=row["targets"], replay=args.replay)
            old.json_write(PRIVATE / "heartbeat.json", beat)
            print(beat, flush=True)
        counts = {k: sum(r[k] for r in output) for k in ("queries", "original_queries", "targets", "complete", "partial", "absent")}
        old.immutable_json(OUT / "analysis.json", dict(registration=reg, result_source="fresh_run",
            counts=counts, summary=summarize(output), record_receipts={r["recording"]: artifact(PRIVATE / f"{r['recording']}.json")
                for r in output}, training=False, original_2pct=True, deployment=False, confirmation_access=False))
        old.json_write(PRIVATE / "heartbeat.json", dict(pid=os.getpid(), seconds=round(time.monotonic() - start, 3),
            phase="complete", recordings=len(output), replay=args.replay, **counts))
        print(read(PRIVATE / "heartbeat.json"), flush=True)


if __name__ == "__main__":
    main()
