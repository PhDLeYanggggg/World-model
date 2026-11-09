"""Test whether more AV2 training scenes improve the additive risk control."""

import argparse
from collections import Counter, defaultdict
import fcntl
from hashlib import sha256
import os
from pathlib import Path
import platform
import sys
from time import time

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import acquire_m3w_av2_development as acquire
from scripts import run_m3w_av2_learned_reference as run
from scripts import audit_m3w_av2_learned_reference as audit

av, prev, np, torch, study, base = run.av, run.prev, run.np, run.torch, run.study, run.base
OUT, LOCAL = acquire.OUT, av.LOCAL.parent/"av2_data_scale_v1"
SEEDS, BATCH, STEPS, SAVE = run.SEEDS, run.BATCH, run.STEPS, run.SAVE
METHODS = ("small", "large")
ARMS = ("cv", "ols4", "fixed_additive", "fixed_joint", "small_point", "large_point",
        "small", "large", "small_released", "large_released")


def beat(phase, **fields):
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(dict(pid=os.getpid(), epoch=time(), phase=phase, **fields))+"\n")
    p.replace(LOCAL/"heartbeat.json")


def register(phase):
    key, old, _, seal, _ = run.register()
    old_s = run.OUT/"summary.json"
    result = study.read(old_s)
    freeze = study.read(study.checked(result["freeze"]))
    _, costs, relatives = av.register()
    manifest = acquire.SOURCE/"manifest.json"
    source = study.read(manifest)
    if (result["registration"] != key or freeze["registration"] != key or freeze["held_labels_used"]
            or result["reserved_outcomes_opened"] or source["registration"] != base.sha(OUT/"data_registration.json")
            or len(source["objects"]) != 1024 or not set(old["files"]) < set(source["objects"])):
        raise ValueError("Need previous verified folds and the fixed UUID expansion")
    p = OUT/"registration.json"
    config = dict(code=[study.artifact(Path(x)) for x in (__file__, run.__file__, av.__file__)],
        source=study.artifact(manifest), old_result=study.artifact(old_s), old_freeze=result["freeze"],
        seeds=SEEDS, observation=av.OBS, future=av.FUT, candidate_roster="same observed 8-step complete moving tracks",
        cohorts="same rotating cities; both arms calibrate and score only added UUIDs, large includes added donor cities only",
        estimand="hold initialization, optimizer updates, candidate menu, seed, peaks, easy/hard cuts, risk and city roles fixed",
        training="same support and point action model, 320 updates, original cutoffs from small donor set",
        calibration="additive-only diagnostic, unchanged dual accuracy and original 2 percent tests, no joint release",
        num_workers=0, torch_threads=4, independent_confirmation=False, reserved_outcomes_opened=False)
    if p.exists() and phase == "score":
        saved = study.read(p)
        if saved["code"][0] != config["code"][0]:
            # Score a sealed fit without changing its registration. The
            # archived training script must match the original receipt.
            archive = LOCAL/"registered_fit_source.py"
            if base.sha(archive) != saved["code"][0]["sha256"]:
                raise ValueError("Cannot verify the registered input and fit code")
            config["code"][0] = saved["code"][0]
    base.immutable(p, config)
    return base.sha(p), old, seal, source, freeze, costs, relatives


def history_fingerprint(h):
    return sha256(b"".join(sorted(row.astype("<f8").tobytes() for row in h))).hexdigest()


@torch.inference_mode()
def freeze_inputs(key, old, seal, source, costs, relatives):
    prior = prev.read_inputs(old, seal)
    models = {s: (av.frozen.load(s, costs, av.turn.refit.model.CostHead, av.turn.refit.STEPS),
                  av.frozen.load(s, relatives, av.slate.model.RelativeGain, av.slate.STEPS)) for s in SEEDS}
    files, duplicates, digest = {}, defaultdict(list), set()
    for n, r in sorted(source["objects"].items()):
        src = study.checked(r["local"])
        if r["local"]["sha256"] in digest:
            raise ValueError("Two UUIDs have identical publisher payloads")
        digest.add(r["local"]["sha256"])
        p = LOCAL/f"{n}_input.npz"
        if not p.exists():
            if n in prior:
                d = prior[n]
            else:
                rows = av.pq.read_table(src, columns=list(av.INPUT_COLUMNS), use_threads=False,
                    filters=[("timestep", "in", list(av.OBS)), ("observed", "=", True)]).to_pylist()
                city, _, ids, h = av.past_roster(rows)
                d = dict(city=city, h=h, ids=ids)
                if len(h) >= 2:
                    cv, ols, _, _ = av.motion.rollout(h, len(av.FUT))
                    tr = av.motion.turn_path(h, len(av.FUT))
                    _, own, _, _ = prev.replay.menu.model.candidates(h, len(av.FUT))
                    bank = np.concatenate((ols[:, None], tr[:, None], own), axis=1)
                    _, _, _, history_gain = run.cost.features(h, cv, ols, bank[:, :2])
                    peaks = av.motion.peak(ols-cv)
                    ok = np.flatnonzero((history_gain > 0.) & (peaks > 1e-9))
                    keep = ok[np.argsort(-history_gain[ok], kind="stable")[:min(2, len(cv)-1)]]
                    anchor = cv.copy()
                    anchor[keep] = ols[keep]
                    d.update(cv=cv, ols4=ols, bank=bank)
                    for s, (c, p_model) in models.items():
                        d[f"anchor_{s}"] = anchor
                        value = prev.replay.choose(h, cv, ols, bank, peaks[keep], c, p_model)
                        d.update({f"{k}_{s}": x for k, x in value.items()})
            ref = av.turn.exp.previous.save_bank(p, dict(**d, registration=key, source=r["local"]["sha256"]))
        else:
            ref = study.artifact(p)
        with np.load(p, allow_pickle=False) as z:
            if z["registration"].item() != key or z["source"].item() != r["local"]["sha256"]:
                raise ValueError("Cannot resume a different input lineage")
            city, h = z["city"].item(), z["h"]
        if len(h) >= 2:
            duplicates[history_fingerprint(h)].append((n, city))
        files[n] = dict(path=ref, city=city, roster=len(h), added=n not in prior)
        if len(files) % 16 == 0:
            beat("freeze_inputs", done=len(files), total=len(source["objects"]))
    same = [g for g in duplicates.values() if len(g) > 1]
    if any(len({city for _, city in g}) != 1 for g in same):
        raise ValueError("Exact observed rosters cross city roles")
    if any(len({files[n]["added"] for n, _ in g}) != 1 for g in same):
        raise ValueError("Added cohort repeats an exact original roster")
    base.immutable(OUT/"inputs.json", dict(registration=key, files=files, exact_duplicate_roster_groups=len(same),
        same_city_extra_rows=sum(len(g)-1 for g in same), input_future_used=False))


def read_inputs(key, source):
    p = OUT/"inputs.json"
    f = study.read(p)
    if f["registration"] != key or f["input_future_used"] or set(f["files"]) != set(source["objects"]):
        raise ValueError("Require complete past-only inputs")
    data = {}
    for n, x in f["files"].items():
        if x["roster"] < 2:
            continue
        with np.load(study.checked(x["path"]), allow_pickle=False) as z:
            d = dict(z)
        if d["registration"].item() != key or d["city"].item() != x["city"]:
            raise ValueError("Wrong model input")
        d.update(city=x["city"], added=x["added"])
        data[n] = d
    return data


def train_large(ident, examples):
    seed, held = ident["seed"], ident["held"]
    p = LOCAL/f"{held}_{seed}.pt"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed+5507)
    net = run.Action()
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    start, trace = 0, []
    if p.exists():
        ck = torch.load(p, weights_only=True, map_location="cpu")
        if ck["identity"] != ident or ck["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Incompatible checkpoint")
        start, trace = ck["step"], ck["loss"]
        rng.bit_generator.state = ck["rng"]
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
    for i in range(start+1, STEPS+1):
        counts = rng.multinomial(BATCH, np.full(len(examples), 1./len(examples)))
        sample = [(z, rng.integers(len(z[0]), size=n)) for z, n in zip(examples, counts, strict=True) if n]
        data = [torch.cat([z[k][j] for z, j in sample]) for k in range(3)]
        value = run.action_loss(net, data)
        opt.zero_grad(set_to_none=True)
        value.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if i % SAVE == 0:
            trace.append(float(value.detach()))
            tmp = p.with_suffix(".pending")
            torch.save(dict(identity=ident, step=i, net=net.state_dict(), opt=opt.state_dict(),
                            rng=rng.bit_generator.state, loss=trace), tmp)
            tmp.replace(p)
            beat("fit_large", fold=f"{held}_{seed}", step=i, total=STEPS)
    return net.eval(), dict(checkpoint=study.artifact(p), donor_queries=len(examples), loss=trace)


def freeze(key, source, parent):
    data, done = read_inputs(key, source), {}
    cities = sorted({d["city"] for d in data.values()})
    for held in cities:
        cal, train = prev.roles(cities, held)
        names = lambda c, added=False: [n for n, d in data.items() if d["city"] in c and (not added or d["added"])]
        donor_y = prev.load_targets(source, data, names(train))
        cal_y = prev.load_targets(source, data, names([cal], added=True))
        cut = prev.donor_cut(data, {n: z for n, z in donor_y.items() if not data[n]["added"]})
        held_names = names([held], added=True)
        for seed in SEEDS:
            k = f"{held}_{seed}"
            p = LOCAL/f"{k}.json"
            ident = dict(registration=key, held=held, calibration=cal, donors=train, seed=seed, cut=cut.tolist())
            control = parent["folds"][k]
            np.testing.assert_array_equal(cut, control["cut"])
            if control["action"]["identity"]["donors"] != train or control["action"]["identity"]["seed"] != seed:
                raise ValueError("Small training roles differ from the registered roles")
            if not p.exists():
                small = audit.load(control["action"], run.Action)
                examples = [run.action_targets(data[n], m, y, v, cut) for n, (y, v) in donor_y.items()
                            if (m := run.menu(data[n], seed, cut)) is not None]
                large, receipt = train_large(ident, examples)
                models = dict(small=small, large=large)
                calibration = {}
                for method, net in models.items():
                    rows = []
                    for n, (y, v) in cal_y.items():
                        _, pred, _, s, _ = run.reference(net, data[n], seed, cut)
                        if s > 0.:
                            rows.append(dict(score=s, strict=run.capacity.eligible_additive(prev.metrics(pred, data[n], y, v, cut)),
                                             add_ok=True))
                    calibration[method] = run.prior.calibration(rows)
                forecasts = {}
                for n in held_names:
                    z = dict(registration=key, seed=seed)
                    for method, net in models.items():
                        point, pred, _, s, _ = run.reference(net, data[n], seed, cut)
                        z.update({method: pred, f"{method}_point": point, f"{method}_score": s})
                    forecasts[n] = av.turn.exp.previous.save_bank(LOCAL/f"{n}_{seed}.npz", z)
                base.immutable(p, dict(identity=ident, forecasts=forecasts, calibration=calibration, large=receipt,
                    small=control["action"], donor_scenarios=len(donor_y), added_calibration=len(cal_y)))
            z = study.read(p)
            if z["identity"] != ident or z["small"] != control["action"] or set(z["forecasts"]) != set(held_names):
                raise ValueError("Resumed receipt disagrees with city or cohort")
            study.checked(z["large"]["checkpoint"])
            for ref in z["forecasts"].values():
                study.checked(ref)
            done[k] = z
            beat("freeze", done=len(done), total=len(cities)*len(SEEDS), fold=k)
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, held_labels_used=False))


def score(key, source):
    data = read_inputs(key, source)
    path = OUT/"freeze.json"
    f = study.read(path)
    expected = {f"{d['city']}_{s}" for d in data.values() for s in SEEDS}
    if f["registration"] != key or f["held_labels_used"] or set(f["folds"]) != expected:
        raise ValueError("Need all held forecasts sealed")
    truth = prev.load_targets(source, data, [n for n, d in data.items() if d["added"]])
    rows, opportunity = [], Counter()
    for n, (y, v) in truth.items():
        d = data[n]
        for seed in SEEDS:
            z = f["folds"][f"{d['city']}_{seed}"]
            with np.load(study.checked(z["forecasts"][n]), allow_pickle=False) as raw:
                if raw["registration"].item() != key or raw["seed"].item() != seed:
                    raise ValueError("Wrong held forecasts")
                p = dict(raw)
            c, o = d["cv"], d["ols4"]
            a, j, paths, _ = prev.candidates(d, seed)
            cut = z["identity"]["cut"]
            vals = dict(cv=c, ols4=o, fixed_additive=a, fixed_joint=j)
            for m in METHODS:
                t = z["calibration"][m]["threshold"]
                vals.update({m: p[m], f"{m}_point": p[f"{m}_point"],
                    f"{m}_released": p[m] if t is not None and p[f"{m}_score"].item() >= t else c})
            measurements = {m: prev.metrics(vals[m], d, y, v, cut) for m in ARMS}
            case = {}
            if paths is not None:
                case["two_slots"] = 1
                for m in ("fixed_additive", *METHODS):
                    ok = run.capacity.eligible_additive(measurements[m])
                    _, q, supports = prev.label_terms(d, paths, vals[m], y, v, cut)
                    _, _, support = prev.label_terms(d, vals[m][None], vals[m], y, v, cut)
                    case.update({f"{m}_eligible": int(ok), f"{m}_easy_support": int(support[0]),
                        f"{m}_paired_ceiling": int(ok and q.any())})
                case["any_easy_support"] = int(supports.any())
                opportunity.update(case)
            direct = av.frontier.pair_reduction(vals["large"][None], vals["small"], y, v)[0]
            rows.append(dict(clip=n, city=d["city"], seed=seed, complete=bool(v.all()), **case,
                metrics=measurements, direct_pair=direct.tolist()))
        if len(rows) % (16*len(SEEDS)) == 0:
            beat("score", done=len(rows)//len(SEEDS), total=len(truth))
    metrics = {}
    for m in ARMS:
        active = [r["metrics"][m]["risk"] for r in rows if r["metrics"][m]["risk"]["selected"]]
        metrics[m] = dict(edits=sum(x["selected"] for x in active), edited=len(active),
            risk_pass=sum(x["supported"] for x in active),
            easy_undefined=sum(x["completion_easy_risk_upper"] is None for x in active),
            easy_over_2=sum((x["known_easy_net_degradation"] or 0.) > .02 for x in active),
            dual_risk_pass=sum(run.capacity.eligible_additive(r["metrics"][m]) for r in rows))
        metrics[m].update({k: av.interval(rows, lambda r: r["metrics"][m][k]) for k in ("gain", "ade", "fde", "hard", "tail")})
    if len({metrics[m]["edits"] for m in ARMS[2:-2]}) != 1:
        raise ValueError("Fixed edit budgets diverged")
    def delta(r, k):
        a, b = [r["metrics"][m][k] for m in METHODS]
        return None if a is None or b is None else (np.asarray(b)-np.asarray(a)).tolist()
    pair = {g: av.interval([r for r in rows if g == "all" or r["complete"]], lambda r: r["direct_pair"])
            for g in ("all", "complete")}
    cal = {m: {k: sum(z["calibration"][m][k] for z in f["folds"].values())
              for k in ("tested", "feasible", "accepted", "local_strict")}
           for m in METHODS}
    by_city = {c: dict(Counter({k: sum(r.get(k, 0) for r in rows if r["city"] == c) for k in opportunity}))
               for c in sorted({r["city"] for r in rows})}
    out = dict(registration=key, freeze=study.artifact(path), inputs=study.artifact(OUT/"inputs.json"), source="fresh_run",
        scoring_code=study.artifact(Path(__file__)),
        new_fits=len(f["folds"]), evaluated_added_scenarios=len(truth),
        complete_added_scenarios=sum(bool(v.all()) for _, v in truth.values()),
        added_roster_actors=sum(len(v) for _, v in truth.values()), added_valid_steps=sum(int(v.sum()) for _, v in truth.values()),
        metrics=metrics, opportunity=dict(opportunity), by_city=by_city, calibration=cal,
        held_paired_delta_vs_small={k: av.interval(rows, lambda r: delta(r, k)) for k in ("gain", "ade", "fde", "hard", "tail")},
        held_direct_pair_native=pair, original_2pct=True, independent_confirmation=False,
        joint_release_evaluated=False, reserved_outcomes_opened=False, deployment_changed=False)
    base.immutable(LOCAL/"rows.json", rows)
    base.immutable(OUT/"summary.json", out)
    print(study.json.dumps({k: out[k] for k in ("new_fits", "evaluated_added_scenarios", "calibration", "opportunity")}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("inputs", "fit", "score"))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for d in (OUT, LOCAL):
        d.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, old, seal, src, parent, costs, relative = register(phase)
        if phase == "inputs":
            freeze_inputs(key, old, seal, src, costs, relative)
        elif phase == "fit":
            freeze(key, src, parent)
        else:
            score(key, src)
        beat(phase+"_complete")


if __name__ == "__main__":
    main()
