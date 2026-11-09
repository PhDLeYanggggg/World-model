"""Replay the frozen donor policies on the enlarged AV2 self-motion menu."""

import argparse
import fcntl
from pathlib import Path
import platform
import sys

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_approach_menu as menu

parent, np, study, base = menu.parent, menu.np, menu.study, menu.base
torch, frozen, slate = parent.torch, parent.frozen, parent.slate
OUT, LOCAL = menu.OUT, menu.LOCAL
ARMS = ("additive", "relative_joint")


@torch.inference_mode()
def choose(h, cv, ols, bank, radii, costs, relatives):
    if not len(radii):
        return dict.fromkeys(ARMS, cv)
    paths, row, col, opts, group = parent.optional.options(cv, bank, radii)
    opts = opts[group == (2 if len(radii) == 2 else 0)]
    x, scale, valid, _ = frozen.shared.costs.features(h, cv, ols, paths)
    if not len(opts) or not valid[row, col].all():
        raise ValueError("Expanded exact-slot menu lost its valid anchor")
    inputs = torch.from_numpy(x[row, col])
    prediction = torch.stack([n(inputs) for n in costs]).mean(0).numpy()*scale[row, None]
    add, joint = parent.turn.refit.model.ranks(prediction, opts)
    a = parent.optional.paint(cv, paths, row, col, opts[int(add.argmax())])
    j = a
    if len(radii) == 2:
        best = -np.inf
        for start in range(0, len(opts), 128):
            trial = slate.assemble(cv, paths, row, col, opts[start:start+128])
            pair = np.mean([slate.gain(n, h, cv, ols, trial, a)[:, 2] for n in relatives], axis=0)
            score = joint[start:start+len(trial)]+slate.WEIGHT*pair
            pick = int(score.argmax())
            if score[pick] > best:
                best, j = score[pick], trial[pick]
    for path in (a, j):
        rr = parent.motion.peak(path-cv)
        np.testing.assert_allclose(np.sort(rr[rr > 0.]), np.sort(radii), atol=1e-8, rtol=1e-8)
    return dict(additive=a, relative_joint=j)


def register():
    key, _, old, source = menu.register()
    p = OUT/"readout.json"
    oracle = study.read(p)
    if oracle["registration"] != key or oracle["reserved_outcomes_opened"]:
        raise ValueError("Require the registered candidate capacity comparison")
    _, costs, relatives = parent.register()
    registration = OUT/"replay_registration.json"
    base.immutable(registration, dict(parent=old["registration"], code=study.artifact(Path(__file__)),
        capacity=study.artifact(p), source=study.artifact(parent.SOURCE/"manifest.json"),
        selection="own-history menu selected for diagnostic after capacity result; not independent",
        scope="all parent scenarios, original matched peaks, no model refit or threshold change",
        arms=ARMS, seeds=slate.SEEDS, original_2pct=True, reserved_outcomes_opened=False))
    return base.sha(registration), old, source, costs, relatives


def freeze(key, old, costs, relatives):
    weights = {s: (frozen.load(s, costs, parent.turn.refit.model.CostHead, parent.turn.refit.STEPS),
                   frozen.load(s, relatives, slate.model.RelativeGain, slate.STEPS)) for s in slate.SEEDS}
    refs = {}
    for name, item in sorted(old["files"].items()):
        with np.load(study.checked(item), allow_pickle=False) as z:
            h, cv, ols = (z[k] for k in ("h", "cv", "ols4"))
            _, own, _, _ = menu.model.candidates(h, len(parent.FUT))
            bank = np.concatenate((ols[:, None], z["turn"][:, None], own), axis=1)
            _, _, _, past = frozen.shared.costs.features(h, cv, ols, bank[:, :2])
            peak = parent.motion.peak(ols-cv)
            eligible = np.flatnonzero((past > 0.) & (peak > 1e-9))
            keep = eligible[np.argsort(-past[eligible], kind="stable")[:min(2, len(cv)-1)]]
            radii = peak[keep]
            out = {}
            for seed, (c, r) in weights.items():
                rr = parent.motion.peak(z[f"anchor_{seed}"]-cv)
                np.testing.assert_array_equal(np.sort(rr[rr > 0.]), np.sort(radii))
                control = choose(h, cv, ols, bank[:, :2], radii, c, r)
                for arm in ARMS:
                    np.testing.assert_array_equal(control[arm], z[f"{arm}_{seed}"])
                for arm, path in choose(h, cv, ols, bank, radii, c, r).items():
                    out[f"{arm}_{seed}"] = path
        refs[name] = parent.turn.exp.previous.save_bank(LOCAL/f"replay_{name}.npz", out)
        menu.beat("replay_freeze", len(refs), len(old["files"]))
    base.immutable(OUT/"replay_freeze.json", dict(registration=key, banks=refs,
        all_parent_predictions_reproduced=True, outcomes_used=False))


def score(key, old, source):
    seal_path = OUT/"replay_freeze.json"
    seal = study.read(seal_path)
    if seal["registration"] != key or seal["outcomes_used"] or set(seal["banks"]) != set(old["files"]):
        raise ValueError("Require the completed forecast freeze")
    data, rows = {}, []
    for name, item in sorted(old["files"].items()):
        with np.load(study.checked(item), allow_pickle=False) as z:
            h = z["h"]
            y, v, _ = parent.targets(study.checked(source["objects"][name]["local"]), z["ids"], h)
            complete = v.all(1)
            e = np.linalg.norm(z["cv"][complete]-y[complete], axis=-1).mean(1)
            data[name] = (z["city"].item(), y, v, e[e > 0.])
    for name, (city, y, v, _) in data.items():
        cut = np.quantile(np.concatenate([e for c, _, _, e in data.values() if c != city]), [.25, .75])
        with (np.load(study.checked(old["files"][name]), allow_pickle=False) as z,
              np.load(study.checked(seal["banks"][name]), allow_pickle=False) as expanded):
            cv, ols = z["cv"], z["ols4"]
            q = np.zeros(len(cv), int)
            den = parent.frontier.denominators(q, v)
            for s in slate.SEEDS:
                by_arm = {}
                for version, bank in (("base", z), ("own", expanded)):
                    a, j = (bank[f"{arm}_{s}"] for arm in ARMS)
                    aa = {}
                    for arm, path in zip(ARMS, (a, j), strict=True):
                        m = study.policy(path, cv, cv, y, v, ["r"]*len(cv), easy_cut=cut[0], hard_cut=cut[1])
                        aa[arm] = dict(gain_cv_pp=frozen.optional_gain(cv, ols, path, y, v, q)[0],
                            ade_complete=float(np.linalg.norm(path[v.all(1)]-y[v.all(1)], axis=-1).mean()) if v.all(1).any() else None,
                            fde=m["available_endpoint_FDE_diagnostic_pixel"], risk=m["recordings"]["r"])
                    effect = parent.frontier.aggregate_pair(j, a, y, v, q, den).tolist() if den.all() else None
                    by_arm[version] = dict(policies=aa, pair=effect)
                rows.append(dict(city=city, clip=name, seed=s, complete=bool(v.all()), arms=by_arm))
        menu.beat("replay_score", len(rows), len(data)*len(slate.SEEDS))
    results = {}
    for ver in ("base", "own"):
        result = results[ver] = {}
        for arm in ARMS:
            take = lambda r: r["arms"][ver]["policies"][arm]
            active = [take(r)["risk"] for r in rows if take(r)["risk"]["selected"]]
            result[arm] = dict(gain_cv_pp=parent.interval(rows, lambda r: take(r)["gain_cv_pp"]),
                ade_complete=parent.interval(rows, lambda r: take(r)["ade_complete"]),
                fde=parent.interval(rows, lambda r: take(r)["fde"]),
                risk_pass=sum(r["supported"] for r in active), edited_views=len(active),
                edits=sum(r["selected"] for r in active),
                signed_easy_over_2pct=sum((r["known_easy_net_degradation"] or 0.) > .02 for r in active))
        for cohort, vals in (("all", rows), ("complete", [r for r in rows if r["complete"]])):
            result[f"pair_{cohort}"] = parent.interval(vals, lambda r: r["arms"][ver]["pair"])
    for arm in ARMS:
        if results["base"][arm]["edits"] != results["own"][arm]["edits"]:
            raise ValueError("Intervention budget changed")
    base.immutable(OUT/"replay_readout.json", dict(registration=key, freeze=study.artifact(seal_path),
        results=results, source="fresh_run", checkpoint_source="cached_verified",
        independent_confirmation=False, reserved_outcomes_opened=False, new_training=False))
    print(study.json.dumps(results, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "score"))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, old, source, costs, relatives = register()
        if phase == "freeze":
            freeze(key, old, costs, relatives)
        else:
            score(key, old, source)
        menu.beat(phase+"_replay_complete", 1, 1)


if __name__ == "__main__":
    main()
