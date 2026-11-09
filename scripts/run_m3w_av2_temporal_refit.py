"""Match donor updates to distinguish menu coverage from score extrapolation."""

import argparse
from collections import Counter
import fcntl
from pathlib import Path
import platform
import sys

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_temporal_menu as tm

np, torch, eq, prev, av, old, base, study, assign = (
    tm.np, tm.torch, tm.eq, tm.prev, tm.av, tm.old, tm.base, tm.study, tm.assign)
OUT, LOCAL = tm.OUT.parent/"av2_temporal_refit_v1", tm.LOCAL.parent/"av2_temporal_refit_v1"
DESIGN = dict(refit_base=("base", "base"), base_on_union=("base", "union"),
              union_on_base=("union", "base"), refit_union=("union", "union"))
ALL = ("old_base", "old_union", *DESIGN)
STEPS, SAVE, BATCH = 320, 40, 96


def beat(phase, **fields):
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(dict(phase=phase, pid=tm.os.getpid(), epoch=tm.time(), **fields))+"\n")
    p.replace(LOCAL/"heartbeat.json")


def register():
    key, data, source, roles, outer, control, parent, _ = tm.register()
    p = tm.OUT/"summary.json"
    r = study.read(p)
    frozen = study.read(study.checked(r["freeze"]))
    _, _, _, _, inner, _, _, _ = assign.register()
    if (r["registration"] != key or frozen["registration"] != key or frozen["held_labels_used"]
            or r["reserved_outcomes_opened"] or set(parent["folds"]) != set(frozen["folds"])):
        raise ValueError("Need the frozen temporal menus and completed development diagnostic")
    reg = OUT/"registration.json"
    base.immutable(reg, dict(code=[study.artifact(Path(__file__)), study.artifact(Path(old.action.__file__))],
        parent=study.artifact(p), parent_freeze=r["freeze"], primary="refit_union",
        design={k: list(v) for k, v in DESIGN.items()}, steps=STEPS, checkpoint=SAVE, batch=BATCH,
        init="same donor-only existing additive weights for both refits", seeds=eq.SEEDS,
        donor_sampling="same scenes and legal actions; one arm receives base, other replaces actions with uniformly drawn temporal shape",
        training="same independent-action regression and edited easy loss; no joint labels or model",
        selection="unchanged independent scoring; both base and union decoding for each refit",
        calibration="all distinct nonempty prefixes under original standalone dual-floor and three 2 percent tests",
        original_2pct=True, num_workers=0, torch_threads=4,
        independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(reg), data, source, roles, inner, outer, control, frozen, r


def examples(data, dy, raw, seed, cut):
    if set(raw) != set(dy):
        raise ValueError("Mismatched donor label roster")
    out = []
    for n, (y, v) in dy.items():
        d = data[n]
        bank = eq.expanded(d, raw[n])
        m, _ = tm.menu(d, seed, cut, raw[n])
        if m is None:
            continue
        width = bank.shape[1]
        family = (m["col"] % (3*width)) // width
        # Match slot, actor and source family exactly across timing variants.
        index = [np.flatnonzero(family == j) for j in range(3)]
        keys = np.column_stack((m["row"], m["col"]//(3*width), m["col"] % width))
        for j in index[1:]:
            np.testing.assert_array_equal(keys[index[0]], keys[j])
        values = old.action.action_targets(d, m, y, v, cut)
        out.append(tuple(torch.stack([x[j] for j in index]) for x in values))
    if not out:
        raise ValueError("Empty temporal fit")
    return out


def fit(ident, name, initial, training):
    nets = torch.nn.ModuleDict({k: old.action.Action() for k in ("base", "union")})
    for net in nets.values():
        net.load_state_dict(initial.state_dict())
    opt = torch.optim.AdamW(nets.parameters(), lr=.001)
    rng = np.random.default_rng(ident["seed"]+6619)
    path = LOCAL/f"{name}.pt"
    start, trace = 0, []
    if path.exists():
        ck = torch.load(path, weights_only=True, map_location="cpu")
        if ck["identity"] != ident or ck["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Invalid temporal checkpoint")
        start, trace = ck["step"], ck["trace"]
        nets.load_state_dict(ck["net"])
        opt.load_state_dict(ck["optimizer"])
        rng.bit_generator.state = ck["rng"]
    for step in range(start+1, STEPS+1):
        count = rng.multinomial(BATCH, np.full(len(training), 1./len(training)))
        chunks = [(r, rng.integers(r[0].shape[1], size=n), rng.integers(3, size=n))
                  for r, n in zip(training, count, strict=True) if n]
        losses = []
        for k, net in nets.items():
            batch = [torch.cat([r[j][0 if k == "base" else f, i] for r, i, f in chunks]) for j in range(3)]
            losses.append(old.action.action_loss(net, batch))
        opt.zero_grad(set_to_none=True)
        sum(losses).backward()
        torch.nn.utils.clip_grad_norm_(nets.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % SAVE == 0:
            trace.append([float(v.detach()) for v in losses])
            tmp = path.with_suffix(".pending")
            torch.save(dict(identity=ident, step=step, trace=trace, net=nets.state_dict(),
                            optimizer=opt.state_dict(), rng=rng.bit_generator.state), tmp)
            tmp.replace(path)
            beat("fit", fold=name, step=step, total=STEPS)
    return nets.eval(), dict(checkpoint=study.artifact(path), trace=trace, donor_queries=len(training))


@torch.inference_mode()
def choices(nets, m, keep):
    x = torch.from_numpy(m["x"])
    z = {k: net(x).numpy() for k, net in nets.items()}
    unit = m["unit"][m["row"]]
    selected, conf = {}, {}
    for k, (training, decode) in DESIGN.items():
        indices = np.flatnonzero(keep[decode])
        _, order, conf[k] = old.action.rank(z[training], m["opts"][indices], unit)
        selected[k] = int(indices[order[0]])
    return selected, conf


def calibrate(nets, forecaster, data, cy, seed, cut):
    raw, offsets = eq.predict(forecaster, data, list(cy))
    rows = {k: [] for k in DESIGN}
    for n, lo, hi in zip(cy, offsets[:-1], offsets[1:], strict=True):
        d, (y, v) = data[n], cy[n]
        m, keep = tm.menu(d, seed, cut, raw[lo:hi])
        if m is not None:
            pick, conf = choices(nets, m, keep)
            for k, p in zip(DESIGN, tm.render(d, m, list(pick.values())), strict=True):
                ok = eq.ceiling.eligible_additive(prev.metrics(p, d, y, v, cut))
                rows[k].append(dict(score=conf[k], strict=ok, add_ok=True))
    return {k: old.action.prior.calibration(rows[k]) for k in DESIGN}


def freeze(key, data, source, roles, inner, outer, control, parent):
    done = {}
    for held, role in roles.items():
        cal, donors = role["calibration"], role["donors"]
        dy = prev.load_targets(source, data, [n for n, d in data.items() if d["city"] in donors])
        cy = prev.load_targets(source, data, [n for n, d in data.items() if d["city"] == cal and d["added"]])
        for seed in eq.SEEDS:
            name = f"{held}_{seed}"
            item, action_ref, f = outer["folds"][name], control["folds"][name], parent["folds"][name]
            cut = f["identity"]["cut"]
            ident = dict(f["identity"], registration=key)
            receipt = LOCAL/f"{name}.json"
            if not receipt.exists():
                raw_donor = old.donor_forecasts(inner, data, held, role, seed)
                initial = assign.prior.load_action(action_ref)
                nets, fit_ref = fit(ident, name, initial, examples(data, dy, raw_donor, seed, cut))
                table = calibrate(nets, old.load_outer(item), data, cy, seed, cut)
                with (np.load(study.checked(item["forecast"]), allow_pickle=False) as raw,
                      np.load(study.checked(f["forecast"]), allow_pickle=False) as prior):
                    for k in ("names", "offsets"):
                        np.testing.assert_array_equal(raw[k], prior[k])
                    picks, scores = {k: [] for k in DESIGN}, {k: [] for k in DESIGN}
                    for i, (n, lo, hi) in enumerate(zip(raw["names"], raw["offsets"][:-1], raw["offsets"][1:], strict=True)):
                        m, keep = tm.menu(data[n], seed, cut, raw["raw_eq"][lo:hi])
                        if (tm.digest(m) if m is not None else "") != prior["digest"][i]:
                            raise ValueError("Temporal menu differs from frozen parent")
                        at, conf = choices(nets, m, keep) if m is not None else (
                            dict.fromkeys(DESIGN, -1), dict.fromkeys(DESIGN, -1.))
                        for k in DESIGN:
                            picks[k].append(at[k])
                            scores[k].append(conf[k])
                    bank = av.turn.exp.previous.save_bank(LOCAL/f"{name}.npz", dict(
                        names=raw["names"], offsets=raw["offsets"], registration=key,
                        **{f"choice_{k}": np.asarray(v) for k, v in picks.items()},
                        **{f"score_{k}": np.asarray(v) for k, v in scores.items()}))
                base.immutable(receipt, dict(identity=ident, fit=fit_ref, forecast=bank, calibration=table))
            r = study.read(receipt)
            if r["identity"] != ident:
                raise ValueError("Invalid refit receipt")
            study.checked(r["forecast"])
            study.checked(r["fit"]["checkpoint"])
            done[name] = r
            beat("freeze", fold=name, completed=len(done), total=len(parent["folds"]))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, held_labels_used=False))


def score(key, data, source, outer, parent, report):
    p = OUT/"freeze.json"
    f = study.read(p)
    if f["registration"] != key or f["held_labels_used"] or set(f["folds"]) != set(parent["folds"]):
        raise ValueError("Require completed temporal refit freeze")
    truth = prev.load_targets(source, data, [n for n, d in data.items() if d["added"]])
    total, cities, rows = {k: Counter() for k in ALL}, {}, []
    for name, item in f["folds"].items():
        held, seed, cut = (item["identity"][k] for k in ("held", "seed", "cut"))
        city = cities.setdefault(held, {k: Counter() for k in ALL})
        ref = parent["folds"][name]
        with (np.load(study.checked(item["forecast"]), allow_pickle=False) as z,
              np.load(study.checked(ref["forecast"]), allow_pickle=False) as old_z,
              np.load(study.checked(outer["folds"][name]["forecast"]), allow_pickle=False) as raw):
            for k in ("names", "offsets"):
                np.testing.assert_array_equal(z[k], old_z[k])
                np.testing.assert_array_equal(z[k], raw[k])
            if z["registration"].item() != key:
                raise ValueError("Invalid registered forecast")
            for i, (n, lo, hi) in enumerate(zip(z["names"], z["offsets"][:-1], z["offsets"][1:], strict=True)):
                d, (y, v) = data[n], truth[n]
                if d["city"] != held:
                    raise ValueError("Held roster mismatch")
                m, keep = tm.menu(d, seed, cut, raw["raw_eq"][lo:hi])
                if m is None:
                    continue
                if tm.digest(m) != old_z["digest"][i]:
                    raise ValueError("Menu digest changed")
                _, _, valid = assign.labels(d, m, y, v, cut)
                valid = valid.numpy()
                ix = {k: int(old_z[f"choice_{k[4:]}"][i] if k.startswith("old_") else z[f"choice_{k}"][i]) for k in ALL}
                pred = tm.render(d, m, list(ix.values()))
                stat, gain = {}, {}
                mass = float(np.where(v, np.linalg.norm(d["cv"]-y, axis=-1), 0.).sum())
                for k, path in zip(ALL, pred, strict=True):
                    decode = k[4:] if k.startswith("old_") else DESIGN[k][1]
                    if not keep[decode][ix[k]]:
                        raise ValueError("Refit forecast outside registered decode bank")
                    met = prev.metrics(path, d, y, v, cut)
                    ok = eq.ceiling.eligible_additive(met)
                    if ok != bool(valid[ix[k]]):
                        raise ValueError("Selected option fails exact label replay")
                    potential = False
                    if ok:
                        pool = tm.render(d, m, np.flatnonzero(valid & keep[decode]))
                        potential = tm.witness(d, pool, path, y, v, cut, mass)
                    receipt, bank, mode = (ref, old_z, decode) if k.startswith("old_") else (item, z, k)
                    t = receipt["calibration"][mode]["threshold"]
                    active = t is not None and bank[f"score_{mode}"][i] >= t
                    stat[k] = dict(eligible=int(ok), pair_available=int(potential), released=int(active),
                        released_eligible=int(active and ok), unsupported=int(met["risk"]["completion_easy_risk_upper"] is None))
                    gain[k] = assign.cost.audit.direct_trajectory(path, pred[0], d["cv"], y, v)
                    total[k].update(stat[k])
                    city[k].update(stat[k])
                rows.append(dict(city=held, clip=str(n), seed=seed, stat=stat, trajectory=gain))
        beat("score", fold=name, two_slot_views=len(rows))
    for k in ("old_base", "old_union"):
        expected = report["counts"][k[4:]]
        if (total[k]["eligible"], total[k]["pair_available"]) != (expected["selected_eligible"], expected["selected_pair"]):
            raise ValueError("Frozen scorer failed to replay")
    if len(rows) != report["two_slot_views"]:
        raise ValueError("Refit changed the roster")
    compare = {"refit_union_vs_base_on_union": ("refit_union", "base_on_union"),
               "refit_union_vs_refit_base": ("refit_union", "refit_base"),
               "refit_union_vs_old_union": ("refit_union", "old_union")}
    out = dict(registration=key, freeze=study.artifact(p), result_source="fresh_run", two_slot_views=len(rows),
        head_fits=2*len(f["folds"]), counts={k: dict(v) for k, v in total.items()},
        by_city={k: {m: dict(v) for m, v in x.items()} for k, x in cities.items()},
        paired_delta={k: {metric: av.interval(rows, lambda r: r["stat"][a][metric]-r["stat"][b][metric])
                          for metric in ("eligible", "pair_available")} for k, (a, b) in compare.items()},
        forced_trajectory_vs_old_base={k: av.interval(rows, lambda r: r["trajectory"][k]) for k in ALL[1:]},
        calibration_accepted={k: sum(v["calibration"][k]["accepted"] for v in f["folds"].values()) for k in DESIGN},
        release_scope="standalone reference diagnostic", original_2pct=True, independent_confirmation=False,
        reserved_outcomes_opened=False, deployment_changed=False)
    base.immutable(OUT/"summary.json", out)
    print(study.json.dumps({k: out[k] for k in ("counts", "paired_delta", "calibration_accepted")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "score"))
    phase = parser.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for d in (LOCAL, OUT):
        d.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, data, source, roles, inner, outer, control, parent, report = register()
        if phase == "fit":
            freeze(key, data, source, roles, inner, outer, control, parent)
        else:
            score(key, data, source, outer, parent, report)
        beat(phase+"_complete")


if __name__ == "__main__":
    main()
