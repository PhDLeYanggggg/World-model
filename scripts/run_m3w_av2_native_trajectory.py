"""Frozen, same-budget 8/12 trajectory development comparison on AV2 train."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
from hashlib import sha256
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from time import time
from urllib.parse import urlencode
from urllib.request import urlopen
from xml.etree import ElementTree

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pyarrow.parquet as pq

from scripts import run_m3w_okutama_frozen_transfer as frozen
from src.evaluation import m3w_interaction_frontier as frontier

np, base, study, turn, torch = frozen.np, frozen.base, frozen.study, frozen.turn, frozen.torch
motion, optional, slate = frozen.source.motion, frozen.source.optional, frozen.slate
OUT = frozen.OUT.parent / "av2_native_trajectory_v1"
LOCAL = frozen.LOCAL.parent.parent / OUT.name
SOURCE = ROOT / "external_data/Argoverse2_mf_train_source_audit"
HOST = "https://argoverse.s3.amazonaws.com/"
PREFIX = "datasets/av2/motion-forecasting/train/"
ID = r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"
KEY = re.compile(re.escape(PREFIX) + rf"({ID})/scenario_\1\.parquet")
OBS, FUT = tuple(range(42, 50)), tuple(range(50, 62))
N_PER_PREFIX = 8
MAX_ROSTER = 8
ARMS = ("cv", "ols4", "damped", "turn", *frozen.ARMS, "scene_uniform")
INPUT_COLUMNS = ("track_id", "timestep", "position_x", "position_y", "observed", "city")


def beat(phase, **info):
    p = LOCAL / "heartbeat.pending"
    p.write_text(study.json.dumps(dict(phase=phase, pid=os.getpid(), epoch=time(), **info))+"\n")
    p.replace(LOCAL/"heartbeat.json")


def get(path, limit):
    with urlopen(HOST+path, timeout=30) as response:
        if not response.geturl().startswith(HOST):
            raise ValueError("Unexpected publisher redirect")
        payload = response.read(limit+1)
        if len(payload) > limit:
            raise ValueError("Publisher object exceeds byte cap")
        return payload, response.headers.get("ETag")


def sample():
    """Choose only by published S3 keys, never by available label length."""
    receipt = {}
    for prefix in "0123456789abcdef":
        query = urlencode({"list-type": 2, "prefix": PREFIX+prefix, "max-keys": 64})
        raw, _ = get("?"+query, 100_000)
        xml = ElementTree.fromstring(raw)
        entries = []
        for row in xml.findall("{*}Contents"):
            key = row.findtext("{*}Key")
            match = KEY.fullmatch(key)
            if match and match[1].startswith(prefix):
                entries.append((match[1], dict(key=key, bytes=int(row.findtext("{*}Size")),
                                               etag=row.findtext("{*}ETag"))))
        if len(entries) < N_PER_PREFIX or len(dict(entries)) != len(entries):
            raise ValueError("Incomplete or duplicated train metadata stratum")
        receipt.update(sorted(entries)[:N_PER_PREFIX])
    return receipt


def register():
    _, _, parent, prior, _, _ = frozen.risk.register()
    costs, relative = parent["fits"], prior["fits"]
    path = OUT/"registration.json"
    base.immutable(path, dict(source=HOST+PREFIX, official_release="https://www.argoverse.org/av2.html",
        sample="first eight scenario parquet UUIDs in each of sixteen first-hex train prefixes",
        observed=OBS, future=FUT, observed_field_required=True, input_columns=INPUT_COLUMNS,
        roster="at most eight complete moving histories: closest pair anchor and nearest neighbors; no category, focal ID or future selection",
        arms=ARMS, fixed_budget="at most two positive past OLS4-hindcast slots, exact peak offsets; same menu",
        scene_uniform="choose one shared OLS4/turn family by past hindcast on anchor recipients",
        training_seeds=slate.SEEDS, donor_costs={k: v["checkpoint"] for k, v in costs.items()},
        donor_relative={k: v["checkpoint"] for k, v in relative.items()},
        code=[study.artifact(Path(p)) for p in (__file__, frozen.__file__, frontier.__file__)],
        easy_reference="positive complete CV ADE quartiles from other cities, scoring only; no risk calibration",
        bootstrap="3000 city-cluster draws after averaging seeds within each scenario",
        support="unaltered 2 percent checks per scenario view; not a recording-level certification",
        roles="publisher-train only; distinct annotated-trajectory development source",
        independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(path), costs, relative


def acquire(key):
    path = SOURCE/"manifest.json"
    if path.exists():
        return
    subprocess.run(["git", "check-ignore", "--quiet", str(path)], cwd=ROOT, check=True)
    if any(p.is_symlink() for p in (SOURCE, *SOURCE.parents)):
        raise ValueError("Refuse symlinked source directory")
    SOURCE.mkdir(parents=True, exist_ok=True)
    selection = SOURCE/"selection.json"
    if not selection.exists():
        base.immutable(selection, dict(registration=key, objects=sample()))
    chosen = study.read(selection)
    if chosen["registration"] != key:
        raise ValueError("Publisher train sample belongs to a different registration")

    def fetch(item):
        name, info = item
        match = KEY.fullmatch(info["key"])
        if match is None or match[1] != name or not 0 < info["bytes"] < 4_000_000:
            raise ValueError("Unexpected source path, identity, or size")
        p = SOURCE/f"{name}.parquet"
        downloaded = p.with_suffix(".json")
        if downloaded.exists():
            existing = study.read(downloaded)
            if existing["object"] != info or study.checked(existing["local"]) != p:
                raise ValueError("Cached publisher object changed")
            return name, existing
        payload, etag = get(info["key"], info["bytes"])
        if len(payload) != info["bytes"] or etag != info["etag"] or payload[:4] != b"PAR1":
            raise OSError("Publisher object does not match registered S3 metadata")
        pending = p.with_suffix(".pending")
        pending.write_bytes(payload)
        pending.replace(p)
        existing = dict(object=info, local=study.artifact(p))
        base.immutable(downloaded, existing)
        return name, existing

    receipts = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for name, receipt in pool.map(fetch, sorted(chosen["objects"].items())):
            receipts[name] = receipt
            beat("acquire", done=len(receipts), total=len(chosen["objects"]))
    if Counter(k[0] for k in receipts) != dict.fromkeys("0123456789abcdef", N_PER_PREFIX):
        raise ValueError("Publisher sample lost a UUID stratum")
    base.immutable(path, dict(registration=key, selection=study.artifact(selection), objects=receipts))


def past_roster(rows):
    """Only complete, moving observed tracks can enter the capped roster."""
    data = {}
    cities = set()
    for row in rows:
        t = row["timestep"]
        if not row["observed"] or t not in OBS:
            raise ValueError("Non-observed state entered AV2 inference")
        cities.add(row["city"])
        track = data.setdefault(row["track_id"], {})
        if t in track:
            raise ValueError("Duplicate AV2 history state")
        track[t] = [row["position_x"], row["position_y"]]
    if len(cities) != 1 or not all(cities):
        raise ValueError("Inconsistent city in observed prefix")
    ids, h = [], []
    for k, tr in sorted(data.items()):
        if len(tr) == len(OBS):
            p = np.asarray([tr[t] for t in OBS], dtype=float)
            if not np.isfinite(p).all():
                raise ValueError("Nonfinite history")
            if np.linalg.norm(np.diff(p, axis=0), axis=1).sum() > 1e-5:
                ids.append(k)
                h.append(p)
    h = np.asarray(h).reshape(-1, len(OBS), 2)
    n = len(h)
    if n > MAX_ROSTER:
        d = np.linalg.norm(h[:, None, -1]-h[None, :, -1], axis=-1)
        np.fill_diagonal(d, np.inf)
        anchor = int(d.min(1).argmin())
        keep = np.r_[anchor, np.argsort(d[anchor], kind="stable")[:MAX_ROSTER-1]]
        ids, h = [ids[j] for j in keep], h[keep]
    return next(iter(cities)), n, np.asarray(ids, dtype=str), h


def freeze(key, costs, relative, source):
    folders = LOCAL/"forecasts"
    folders.mkdir(parents=True, exist_ok=True)
    save = turn.exp.previous.save_bank
    nets = {s: (frozen.load(s, costs, turn.refit.model.CostHead, turn.refit.STEPS),
                frozen.load(s, relative, slate.model.RelativeGain, slate.STEPS)) for s in slate.SEEDS}
    files = {}
    for name, receipt in source["objects"].items():
        path = folders/f"{name}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                if z["registration"].item() != key or z["source"].item() != receipt["local"]["sha256"]:
                    raise ValueError("Partial forecast cache lineage mismatch")
            files[name] = study.artifact(path)
            continue
        src = study.checked(receipt["local"])
        rows = pq.read_table(src, columns=list(INPUT_COLUMNS), use_threads=False,
                             filters=[("timestep", "in", list(OBS)), ("observed", "=", True)]).to_pylist()
        city, n, ids, h = past_roster(rows)
        values = dict(city=city, eligible=n, ids=ids, h=h, registration=key, source=receipt["local"]["sha256"])
        if len(h) >= 2:
            cv, ols, other, _ = motion.rollout(h, len(FUT))
            tr = motion.turn_path(h, len(FUT))
            values.update(cv=cv, ols4=ols, damped=other[:, 2], turn=tr)
            _, _, menu = motion.make_menu(h, len(FUT), np.ones(len(h)))
            past = motion.prefix_rank(h, np.ones(len(h)), np.ones(len(h), bool)).mean(0)
            for seed, (c, r) in nets.items():
                pred = frozen.predict(h, np.zeros(len(h), int), cv, ols, tr, c, r)
                active = motion.peak(pred["anchor"]-cv) > 0
                uniform = cv.copy()
                if active.any():
                    idx = (0, -1)[int(np.argmax(past[active][:, [0, -1]].sum(0)))]
                    uniform[active] = menu[active, idx]
                np.testing.assert_allclose(np.sort(motion.peak(uniform-cv)),
                                           np.sort(motion.peak(pred["anchor"]-cv)), rtol=1e-8, atol=1e-8)
                values.update({f"{a}_{seed}": x for a, x in dict(pred, scene_uniform=uniform).items()})
        files[name] = save(path, values)
        beat("freeze", done=len(files), total=len(source["objects"]))
    base.immutable(OUT/"freeze.json", dict(registration=key, files=files, future_rows_used=False,
                                          model_source="cached_verified", complete_before_scoring=True))


def targets(path, ids, h):
    y = np.full((len(h), len(FUT), 2), np.nan)
    lookup = {k: i for i, k in enumerate(ids)}
    cols = ("track_id", "timestep", "position_x", "position_y", "object_type")
    rows = pq.read_table(path, columns=list(cols), use_threads=False,
                         filters=[("timestep", "in", [OBS[-1], *FUT]), ("track_id", "in", ids.tolist())]).to_pylist()
    types, seen = {}, set()
    for row in rows:
        t, k = row["timestep"], row["track_id"]
        if (k, t) in seen:
            raise ValueError("Duplicate AV2 scoring state")
        seen.add((k, t))
        i = lookup[k]
        if t == OBS[-1]:
            types[k] = row["object_type"]
        else:
            y[i, t-FUT[0]] = [row["position_x"], row["position_y"]]
    if set(types) != set(ids):
        raise ValueError("Scoring roster no longer matches the frozen origin")
    y -= h[:, -1:]
    valid = np.isfinite(y).all(-1)
    return np.where(valid[..., None], y, 0.), valid, np.asarray([types[k] for k in ids])


def ceiling(cv, ols, tr, anchor, additive, y, v, cut):
    """Exact two-slot hindsight feasibility, not a learnable policy result."""
    radii = motion.peak(anchor-cv)
    if np.count_nonzero(radii) != 2 or not v.all():
        return None
    paths, row, arm, opts, group = optional.options(cv, np.stack((ols, tr), axis=1), radii[radii > 0])
    bank = slate.assemble(cv, paths, row, arm, opts[group == 2])
    pair = frontier.pair_reduction(bank, additive, y, v)[:, 1]
    errors = np.linalg.norm(bank-y, axis=-1).sum((1, 2))
    floors = [np.linalg.norm(p-y, axis=-1).sum() for p in (cv, ols, additive)]
    want = (pair > 1e-10) & (errors < min(floors[:2])-1e-10) & (errors <= floors[2]+1e-10)
    best = 0.
    for i in np.argsort(-pair):
        if want[i] and study.policy(bank[i], cv, cv, y, v, ["r"]*len(cv),
                                   easy_cut=cut[0], hard_cut=cut[1])["recordings"]["r"]["supported"]:
            best = float(pair[i])
            break
    return dict(relaxed=float(pair.max()), strict=best, candidates=len(bank))


def interval(rows, field):
    cities = sorted({a["city"] for a in rows})
    result = turn.effects.interval([
        turn.effects.site_means([(r["clip"], field(r)) for r in rows if r["city"] == c and field(r) is not None])
        for c in cities])
    if result is not None:
        result["cities"] = result.pop("sites")
        result["resampling_unit"] = "city"
    return result


def summary(rows, arms, sub):
    data = [r for r in rows if sub in r["arms"]]
    pair = interval(data, lambda r: r["pair"][sub])
    result = {}
    for arm in arms:
        metrics = [x["arms"][sub][arm] for x in data]
        changed = [m for m in metrics if m["risk"]["selected"]]
        result[arm] = dict(
            ade_complete=interval(data, lambda r: r["arms"][sub][arm]["ade_complete"]),
            fde_available=interval(data, lambda r: r["arms"][sub][arm]["fde_available"]),
            gain_pp=interval(data, lambda r: r["arms"][sub][arm]["gain"]),
            gain_vs_ols4_pp=interval(data, lambda r: r["arms"][sub][arm]["gain_vs_ols4"]),
            hard_gain_pp=interval(data, lambda r: r["arms"][sub][arm]["hard"]),
            worst_view_p95_known_ade=max((m["p95"] for m in metrics if m["p95"] is not None), default=None),
            edits=sum(m["risk"]["selected"] for m in changed), edited_views=len(changed),
            original_risk_pass=sum(m["risk"]["supported"] for m in changed),
            easy_undefined=sum(m["risk"]["completion_easy_risk_upper"] is None for m in changed),
            signed_easy_over_2pct=sum((m["risk"]["known_easy_net_degradation"] or 0.) > .02 for m in metrics))
    return dict(views=len(data), arms=result, relative_vs_additive_pair_native=pair)


def score(key, source):
    path = OUT/"freeze.json"
    seal = study.read(path)
    if (seal["registration"] != key or seal["future_rows_used"] or not seal["complete_before_scoring"]
            or set(seal["files"]) != set(source["objects"])):
        raise ValueError("Require all frozen predictions before scoring")
    data, support = {}, {}
    for name, receipt in source["objects"].items():
        with np.load(study.checked(seal["files"][name]), allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        city, h = d["city"].item(), d["h"]
        s = support[name] = dict(city=city, eligible=d["eligible"].item(), roster=len(h))
        if len(h) < 2:
            continue
        y, v, ty = targets(study.checked(receipt["local"]), d["ids"], h)
        complete = v.all(1)
        e = np.linalg.norm(d["cv"]-y, axis=-1)
        s.update(complete=int(complete.sum()), endpoint=int(v[:, -1].sum()), grid=int(v.sum()),
                 types=dict(Counter(ty.tolist())), pedestrian_complete=int(np.count_nonzero(complete & (ty == "pedestrian"))))
        data[name] = (city, d, y, v, ty, e[complete].mean(1))
    cases, ceilings = [], []
    for name, (city, d, y, v, ty, _) in data.items():
        cuts = np.concatenate([e[e > 0] for c, _, _, _, _, e in data.values() if c != city])
        cut = np.quantile(cuts, [.25, .75])
        for seed in slate.SEEDS:
            paths = {a: d[a if a in ARMS[:4] else f"{a}_{seed}"] for a in ARMS}
            p = {a: paths[a] for a in frozen.ARMS}
            c = ceiling(d["cv"], d["ols4"], d["turn"], p["anchor"], p["additive"], y, v, cut)
            if c is not None:
                ceilings.append(dict(city=city, clip=name, seed=seed, **c))
            metrics, pair = {}, {}
            for group, at in (("all", np.ones(len(y), bool)), ("pedestrian", ty == "pedestrian")):
                if not at.any():
                    continue
                cv, ols, yy, vv = (a[at] for a in (d["cv"], d["ols4"], y, v))
                q = np.zeros(len(cv), dtype=int)
                by_arm = metrics[group] = {}
                for a, pp in paths.items():
                    pp = pp[at]
                    m = study.policy(pp, cv, cv, yy, vv, ["r"]*len(cv), easy_cut=cut[0], hard_cut=cut[1])
                    error = np.linalg.norm(pp-yy, axis=-1)
                    by_arm[a] = dict(risk=m["recordings"]["r"], gain=frozen.optional_gain(cv, ols, pp, yy, vv, q),
                        gain_vs_ols4=frozen.optional_gain(ols, cv, pp, yy, vv, q),
                        p95=m["p95_known_ADE_pixel"], hard=m["subsets"]["hard"]["gain_percent"],
                        ade_complete=float(error[vv.all(1)].mean()) if vv.all(1).any() else None,
                        fde_available=m["available_endpoint_FDE_diagnostic_pixel"])
                den = frontier.denominators(q, vv)
                pair[group] = (frontier.aggregate_pair(p["relative_joint"][at], p["additive"][at], yy, vv, q, den).tolist()
                               if den.all() else None)
            if len({metrics["all"][a]["risk"]["selected"] for a in ARMS[4:]}) != 1:
                raise ValueError("Arms do not have identical realized intervention counts")
            cases.append(dict(city=city, clip=name, seed=seed, arms=metrics, pair=pair))
        beat("score", scenario=name, cases=len(cases))
    totals = {k: summary(cases, ARMS, k) for k in ("all", "pedestrian")}
    base.immutable(OUT/"readout.json", dict(registration=key, freeze=study.artifact(path), result_source="fresh_run",
        sample_support=support, totals=totals, exact_complete_two_slot_ceiling=dict(views=len(ceilings),
            relaxed_positive=sum(r["relaxed"] > 1e-10 for r in ceilings), strict_positive=sum(r["strict"] > 0 for r in ceilings),
            unique_strict_scenarios=len({r["clip"] for r in ceilings if r["strict"] > 0})),
        per_city={c: summary([r for r in cases if r["city"] == c], ARMS, "all") for c in sorted({r["city"] for r in cases})},
        per_scenario=cases, ceiling=ceilings, coordinates="AV2 native; not pooled with image coordinates",
        independent_confirmation=False, reserved_outcomes_opened=False, no_training_or_threshold_search=True,
        stage5c_executed=False, smc_enabled=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("acquire", "freeze", "score"))
    phase = parser.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    LOCAL.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, costs, relative = register()
        if phase == "acquire":
            acquire(key)
        else:
            source = study.read(SOURCE/"manifest.json")
            if source["registration"] != key:
                raise ValueError("Source receipt does not match registration")
            study.checked(source["selection"])
            if phase == "freeze":
                freeze(key, costs, relative, source)
            else:
                score(key, source)
        beat(phase+"_complete", registration=key)


if __name__ == "__main__":
    main()
