"""Attribute the fixed-reference ceiling to timing, path family, and recipients."""

from collections import Counter
from pathlib import Path
import platform
import sys

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_incumbent_timing as run

np, torch, av, eq, prev, study, base = run.np, run.torch, run.av, run.eq, run.prev, run.study, run.base
tm = run.refit.tm
MODES = ("local_timing", "same_actor_and_peak", "same_actors", "all_recipients")


def locality(actors, radii, peaks):
    peaks = np.asarray(peaks)
    edited = np.flatnonzero(peaks > 0.)
    if actors.ndim != 2 or actors.shape[1] != 2 or len(edited) != 2:
        raise ValueError("Exactly two assigned slots required")
    same = (np.sort(actors, axis=1) == edited).all(1)
    assigned = np.isclose(peaks[actors], np.asarray(radii)[None], rtol=1e-8, atol=1e-8).all(1)
    return same & assigned, same


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, data, source, _, _, outer, control, _, _ = run.register()
    rpath, tpath = run.OUT/"summary.json", tm.OUT/"summary.json"
    r, t = study.read(rpath), study.read(tpath)
    frozen = study.read(study.checked(r["freeze"]))
    if frozen["registration"] != key or frozen["held_labels_used"] or r["reserved_outcomes_opened"] or t["reserved_outcomes_opened"]:
        raise ValueError("Require sealed and completed development results")
    truth = prev.load_targets(source, data, [n for n, d in data.items() if d["added"]])
    total, cities, rows = Counter(), {}, []
    for name, f in frozen["folds"].items():
        held, seed, cut = (f["identity"][k] for k in ("held", "seed", "cut"))
        city = cities.setdefault(held, Counter())
        with (np.load(study.checked(outer["folds"][name]["forecast"]), allow_pickle=False) as raw,
              np.load(study.checked(control["folds"][name]["forecast"]), allow_pickle=False) as a):
            for k in ("names", "offsets"):
                np.testing.assert_array_equal(raw[k], a[k])
            for n, lo, hi in zip(a["names"], a["offsets"][:-1], a["offsets"][1:], strict=True):
                d, (y, v), anchor = data[n], truth[n], a["additive"][lo:hi]
                if d["city"] != held:
                    raise ValueError("Wrong held city")
                if len(eq.radii(d, seed)) != 2:
                    continue
                eligible = eq.ceiling.eligible_additive(prev.metrics(anchor, d, y, v, cut))
                stat = Counter(two_slots=1, anchor_eligible=int(eligible))
                if eligible:
                    m, _ = tm.menu(d, seed, cut, raw["raw_eq"][lo:hi])
                    _, _, safe = tm.assign.labels(d, m, y, v, cut)
                    ix = np.flatnonzero(safe.numpy())
                    pool = tm.render(d, m, ix)
                    c = d["cv"]
                    sub = locality(m["row"][m["opts"][ix]], m["radii"], av.motion.peak(anchor-c))
                    mass = float(np.where(v, np.linalg.norm(c-y, axis=-1), 0.).sum())
                    p = run.prepare(d, anchor, cut)
                    stat["local_timing"] = int(p is not None and run.targets(d, p, anchor, y, v, cut)[:, 0].any())
                    for k, candidate in zip(MODES[1:], (pool[sub[0]], pool[sub[1]], pool), strict=True):
                        stat[k] = int(tm.witness(d, candidate, anchor, y, v, cut, mass))
                    if any(stat[a] > stat[b] for a, b in zip(MODES, MODES[1:])):
                        raise ValueError("Wider union lost a local replacement witness")
                total.update(stat)
                city.update(stat)
                rows.append(dict(clip=str(n), city=held, seed=seed, stat=dict(stat)))
    if (total["local_timing"] != r["counts"]["local_available"]
            or total["all_recipients"] != t["counts"]["union"]["fixed_pair"]):
        raise ValueError("Fixed-reference bounds do not replay")
    out = dict(code=study.artifact(Path(__file__)), input=[study.artifact(p) for p in (rpath, tpath)],
        counts={k: total[k] for k in ("two_slots", "anchor_eligible", *MODES)},
        by_city={c: {k: x[k] for k in MODES} for c, x in cities.items()},
        increments={b: av.interval(rows, lambda r: r["stat"].get(b, 0)-r["stat"].get(a, 0))
                    for a, b in zip(MODES, MODES[1:])},
        result_source="fresh_run", hindsight_only=True, original_2pct=True,
        reserved_outcomes_opened=False, independent_confirmation=False)
    base.immutable(run.OUT/"capacity_attribution.json", out)
    print(study.json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
