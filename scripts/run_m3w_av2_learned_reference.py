"""Learn a risk-aware additive control, then test matched joint re-ranking."""

import argparse
from collections import Counter
import fcntl
from math import log
import os
from pathlib import Path
import platform
import sys
from time import time

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_supported_gain as prior
from scripts import audit_m3w_av2_paired_release_capacity as capacity
from src.evaluation.m3w_cv_anchor_budget import candidate_terms, risk_excess

prev, av, np, torch, F, study, base = prior.prev, prior.av, prior.np, prior.torch, prior.F, prior.study, prior.base
cost = av.frozen.shared.costs
OUT, LOCAL = prev.OUT.parent/"av2_learned_reference_v1", prev.LOCAL.parent/"av2_learned_reference_v1"
SEEDS, STEPS, SAVE, BATCH = prev.SEEDS, prev.STEPS, prev.SAVE, prev.BATCH
TOP = 8
ARMS = ("cv", "ols4", "fixed_additive", "fixed_joint", "point", "support", "joint", "released")


def beat(phase, **fields):
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(dict(pid=os.getpid(), epoch=time(), phase=phase, **fields))+"\n")
    p.replace(LOCAL/"heartbeat.json")


def register():
    parent, old, source, seal, _, previous = prior.register()
    p = prior.OUT/"paired_release_capacity.json"
    r = study.read(p)
    report = study.read(study.checked(r["source"]))
    frozen = study.read(study.checked(report["freeze"]))
    if (report["registration"] != parent or frozen["registration"] != parent or frozen["held_labels_used"]
            or r["reserved_outcomes_opened"] or r["independent_confirmation"]):
        raise ValueError("Need the exact registered development menu and ceiling")
    reg = OUT/"registration.json"
    base.immutable(reg, dict(parent=r["source"], ceiling=study.artifact(p), parent_freeze=report["freeze"],
        code=[study.artifact(Path(x)) for x in (__file__, prev.__file__, prior.__file__, cost.__file__)],
        source="same 128 AV2 train scenarios, native 8/12 grid and original exact peak slots",
        roles="four city donors, one calibration, one held; inner leave-donor-city-out reference for pair supervision",
        arms=ARMS, seeds=SEEDS, steps=STEPS, checkpoint_every=SAVE, top=TOP,
        action="per-action signed CV and OLS increments, all/easy excess and easy signed change; context-only easy probability",
        point="sum mean predicted two-reference increments",
        support="prefer legal assignments with a predicted easy recipient (p>=.5) when available, rank by sum of risk-penalized increments",
        joint="same top-eight support-ranked options; rank conditional qualifying pair gain relative to learned additive",
        release="original two-arm dual reference and 2 percent checks at all nonempty calibration prefixes",
        auxiliary="additive-only calibration is diagnostic and not joint release",
        workers=0, torch_threads=4, independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(reg), old, source, seal, previous


class Action(torch.nn.Module):
    def __init__(self):
        super().__init__()
        nn = torch.nn
        self.value = nn.Sequential(nn.Linear(cost.WIDTH+1, 48), nn.SiLU(), nn.Linear(48, 24), nn.SiLU(), nn.Linear(24, 5))
        self.easy = nn.Sequential(nn.Linear(21, 24), nn.SiLU(), nn.Linear(24, 1))

    def forward(self, x):
        # First 20 bridge features are independent of the candidate trajectory.
        q = self.easy(torch.cat((x[..., :20], x[..., -1:]), dim=-1))
        return torch.cat((self.value(x), q), dim=-1)


def menu(d, seed, cut):
    c, o = d["cv"], d["ols4"]
    radii = av.motion.peak(d[f"anchor_{seed}"]-c)
    radii = radii[radii > 0.]
    if len(radii) != 2:
        return None
    bank, row, col, opts, group = av.optional.options(c, d["bank"], radii)
    x, unit, legal, _ = cost.features(d["h"], c, o, bank)
    if not legal[row, col].all():
        raise ValueError("Exact menu contains a collapsed action")
    x = np.column_stack((x[row, col], np.log1p(cut[0]/unit[row]))).astype(np.float32)
    return dict(bank=bank, row=row, col=col, opts=opts[group == 2], unit=unit, x=x, radii=radii)


def action_targets(d, m, y, v, cut):
    c, o = d["cv"], d["ols4"]
    row, col, b = (m[k] for k in ("row", "col", "bank"))
    _, _, gain, _ = candidate_terms(c, o, b, y, v)
    risk = risk_excess(c, b, y, v, cut[0])
    base_err = np.where(v, np.linalg.norm(c-y, axis=-1), 0.).sum(1)/v.sum(1).clip(1)
    err = np.where(v[:, None], np.linalg.norm(b-y[:, None], axis=-1), 0.).sum(2)/v.sum(1).clip(1)[:, None]
    easy = v.any(1) & (base_err > 0.) & (base_err <= cut[0])
    signed = np.where(easy[row], err[row, col]-base_err[row], 0.)
    signed -= .02*np.where(easy[row], base_err[row], 0.)
    target = np.column_stack((gain[row, col]/v.shape[1], risk[row, col], signed))/m["unit"][row, None]
    return tuple(torch.from_numpy(z.astype(np.float32)) for z in (m["x"], target, easy[row]))


def action_loss(net, batch):
    x, y, easy = batch
    z = net(x)
    return F.smooth_l1_loss(z[:, :5], y, beta=.05)+.25*F.binary_cross_entropy_with_logits(z[:, 5], easy)


def pair_loss(net, batch):
    x, b, g, strict, support = batch
    z = net(x)
    return (prior.gain_loss(z[:, 0], b, g, strict, "conditional")
        +F.binary_cross_entropy_with_logits(z[:, 1], strict, pos_weight=torch.tensor(5.))
        +.25*F.binary_cross_entropy_with_logits(z[:, 2], support))


def fit(identity, examples, network, loss, name):
    p = LOCAL/f"{name}.pt"
    seed = identity["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed+5507)
    net = network()
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    start, trace = 0, []
    if p.exists():
        ck = torch.load(p, weights_only=True, map_location="cpu")
        if ck["identity"] != identity or ck["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Incompatible checkpoint")
        start, trace = ck["step"], ck["loss"]
        rng.bit_generator.state = ck["rng"]
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
    if not examples:
        raise ValueError("Empty donor role")
    for i in range(start+1, STEPS+1):
        counts = rng.multinomial(BATCH, np.full(len(examples), 1./len(examples)))
        sample = [(z, rng.integers(len(z[0]), size=n)) for z, n in zip(examples, counts, strict=True) if n]
        data = [torch.cat([z[k][j] for z, j in sample]) for k in range(len(examples[0]))]
        value = loss(net, data)
        opt.zero_grad(set_to_none=True)
        value.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if i % SAVE == 0:
            trace.append(float(value.detach()))
            tmp = p.with_suffix(".pending")
            torch.save(dict(identity=identity, step=i, net=net.state_dict(), opt=opt.state_dict(),
                            rng=rng.bit_generator.state, loss=trace), tmp)
            tmp.replace(p)
            beat("fit", fold=name, step=i, total=STEPS)
    return net.eval(), dict(identity=identity, checkpoint=study.artifact(p), loss=trace, donor_queries=len(examples))


def rank(z, opts, unit):
    """Independent action scores with a common two-slot support constraint."""
    utility = z[:, :2].mean(1)*unit
    penalty = np.maximum(z[:, 2:5], 0.).sum(1)*unit
    point = utility[opts].sum(1)
    value = (utility-penalty)[opts].sum(1)
    p = 1./(1.+np.exp(-np.clip(z[:, 5], -50., 50.)))
    supported = p[opts].max(1) >= .5
    eligible = np.flatnonzero(supported if supported.any() else np.ones(len(opts), dtype=bool))
    order = eligible[np.argsort(-value[eligible], kind="stable")[:TOP]]
    score = float((1.-np.prod(1.-p[opts[order[0]]]))/(1.+np.exp(-np.clip(value[order[0]]/unit.mean(), -50., 50.))))
    return int(point.argmax()), order, score


@torch.inference_mode()
def reference(net, d, seed, cut):
    m = menu(d, seed, cut)
    if m is None:
        return d[f"additive_{seed}"], d[f"additive_{seed}"], None, -1., None
    z = net(torch.from_numpy(m["x"])).numpy()
    point, rank_ix, confidence = rank(z, m["opts"], m["unit"][m["row"]])
    paths = av.slate.assemble(d["cv"], m["bank"], m["row"], m["col"], m["opts"])
    radii = np.sort(m["radii"])
    np.testing.assert_allclose(np.sort(av.motion.peak(paths-d["cv"]), axis=1)[:, -2:],
        np.broadcast_to(radii, (len(paths), 2)), rtol=1e-8, atol=1e-8)
    return paths[point], paths[rank_ix[0]], paths, confidence, rank_ix


def pair_features(d, paths, ref):
    x, _, b, unit, _ = prev.relative.features(d["h"], d["cv"], d["ols4"], paths, ref)
    return np.column_stack((x, np.log1p(b))).astype(np.float32), b, unit


def pair_example(d, seed, y, v, cut, inner):
    _, ref, paths, _, _ = reference(inner[0], d, seed, inner[1])
    if paths is None:
        return None
    x, b, unit = pair_features(d, paths, ref)
    gain, strict, support = prev.label_terms(d, paths, ref, y, v, cut)
    return tuple(torch.from_numpy(z.astype(np.float32)) for z in (x, b, gain/unit, strict, support))


@torch.inference_mode()
def predict(action, joint, d, seed, cut):
    point, ref, paths, conf, order = reference(action, d, seed, cut)
    if paths is None or joint is None:
        return point, ref, ref, conf, -1.
    x, b, _ = pair_features(d, paths[order], ref)
    pred = joint(torch.from_numpy(x))
    score = pred[:, 0].sigmoid()*torch.from_numpy(b)*(pred[:, 1]-log(5.)).sigmoid()
    k = int(score.argmax())
    return point, ref, paths[order[k]], conf, float(score[k])


def train(key, old, source, seal, parent):
    data, done = prev.read_inputs(old, seal), {}
    cities = sorted({d["city"] for d in data.values()})
    cache = {}

    def names(group):
        return [n for n, d in data.items() if d["city"] in group]

    def anchor(group, seed, truth):
        group = tuple(sorted(group))
        if set(group) != {data[n]["city"] for n in truth}:
            raise ValueError("Action targets cross a registered training role")
        k = (group, seed)
        if k not in cache:
            cut = prev.donor_cut(data, truth)
            ex = [action_targets(data[n], m, y, v, cut) for n, (y, v) in truth.items()
                  if (m := menu(data[n], seed, cut)) is not None]
            ident = dict(registration=key, model="action", donors=list(group), seed=seed, cut=cut.tolist())
            net, item = fit(ident, ex, Action, action_loss, f"action_{'+'.join(group)}_{seed}")
            cache[k] = (net, cut, item)
        return cache[k]

    for held in cities:
        cal, donors = prev.roles(cities, held)
        truth = prev.load_targets(source, data, names(donors))
        cal_y = prev.load_targets(source, data, names([cal]))
        for seed in SEEDS:
            k = f"{held}_{seed}"
            ident = dict(registration=key, held=held, calibration=cal, donors=donors, seed=seed)
            receipt = LOCAL/f"{k}.json"
            if not receipt.exists():
                action, cut, item = anchor(donors, seed, truth)
                np.testing.assert_array_equal(cut, parent["folds"][k]["cut"])
                inner = {city: anchor([g for g in donors if g != city], seed,
                    {n: v for n, v in truth.items() if data[n]["city"] != city}) for city in donors}
                examples = [z for n, (y, v) in truth.items()
                    if (z := pair_example(data[n], seed, y, v, cut, inner[data[n]["city"]])) is not None]
                positives = int(sum(x[3].sum().item() for x in examples))
                if positives:
                    joint, pair_item = fit(dict(ident, model="joint"), examples, prev.network, pair_loss, f"joint_{k}")
                else:
                    joint, pair_item = None, None
                rows, add_rows = [], []
                for n, (y, v) in cal_y.items():
                    d = data[n]
                    _, a, j, sa, sj = predict(action, joint, d, seed, cut)
                    if sa <= 0.:
                        continue
                    ok = capacity.eligible_additive(prev.metrics(a, d, y, v, cut))
                    _, qual, _ = prev.label_terms(d, j[None], a, y, v, cut)
                    add_rows.append(dict(score=sa, strict=ok, add_ok=True))
                    if sj > 0.:
                        rows.append(dict(score=sj, strict=bool(qual[0]), add_ok=ok))
                banks = {}
                for n in names([held]):
                    point, a, j, sa, sj = predict(action, joint, data[n], seed, cut)
                    banks[n] = av.turn.exp.previous.save_bank(LOCAL/f"{n}_{seed}.npz",
                        dict(registration=key, seed=seed, point=point, support=a, joint=j, confidence=sa, score=sj))
                base.immutable(receipt, dict(identity=ident, cut=cut.tolist(), action=item, pair=pair_item,
                    inner={g: m[2] for g, m in inner.items()}, pair_train_positives=positives,
                    calibration=prior.calibration(rows), diagnostic_add_cal=prior.calibration(add_rows), forecasts=banks))
            r = study.read(receipt)
            if r["identity"] != ident or set(r["forecasts"]) != set(names([held])) or set(r["inner"]) != set(donors):
                raise ValueError("Reference training receipt changed")
            for g, nested in r["inner"].items():
                if set(nested["identity"]["donors"]) != set(donors)-{g}:
                    raise ValueError("Nested producer saw its own donor label or the outer held/calibration city")
                study.checked(nested["checkpoint"])
            for ref in (*r["forecasts"].values(), r["action"]["checkpoint"]):
                study.checked(ref)
            if r["pair"] is not None:
                study.checked(r["pair"]["checkpoint"])
            done[k] = r
            beat("freeze", fold=k, done=len(done), total=len(cities)*len(SEEDS))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, held_labels_used=False))


def evaluate(key, old, source, seal, parent):
    p = OUT/"freeze.json"
    frozen = study.read(p)
    if frozen["registration"] != key or frozen["held_labels_used"] or set(frozen["folds"]) != set(parent["folds"]):
        raise ValueError("All forecasts and calibration rules must be frozen")
    data = prev.read_inputs(old, seal)
    truth, rows, counts = prev.load_targets(source, data, data), [], Counter()
    for n, d in data.items():
        y, v = truth[n]
        for seed in SEEDS:
            z = frozen["folds"][f"{d['city']}_{seed}"]
            with np.load(study.checked(z["forecasts"][n]), allow_pickle=False) as raw:
                if raw["registration"].item() != key or raw["seed"].item() != seed:
                    raise ValueError("Wrong held forecast")
                r = dict(raw)
            a, j, paths, _ = prev.candidates(d, seed)
            c, o, cut = d["cv"], d["ols4"], z["cut"]
            t = z["calibration"]["threshold"]
            release = r["joint"] if t is not None and r["score"].item() >= t else c
            pred = dict(zip(ARMS, (c, o, a, j, r["point"], r["support"], r["joint"], release), strict=True))
            m = {k: prev.metrics(q, d, y, v, cut) for k, q in pred.items()}
            case = {}
            if paths is not None:
                counts["two_slots"] += 1
                for mode in ("fixed_additive", "point", "support"):
                    _, q, _ = prev.label_terms(d, paths, pred[mode], y, v, cut)
                    legal = capacity.eligible_additive(m[mode])
                    case.update({f"{mode}_eligible": int(legal), f"{mode}_paired_ceiling": int(legal and q.any())})
                _, q, _ = prev.label_terms(d, pred["joint"][None], pred["support"], y, v, cut)
                case.update(joint_strict=int(q[0]), joint_and_support=int(q[0] and case["support_eligible"]))
                counts.update(case)
            term = {f"{a}_vs_{b}": av.frontier.pair_reduction(pred[a][None], pred[b], y, v)[0].tolist()
                    for a, b in (("joint", "support"), ("support", "fixed_additive"), ("joint", "fixed_joint"))}
            rows.append(dict(clip=n, city=d["city"], seed=seed, complete=bool(v.all()), metrics=m, pair=term, **case))
        beat("score", done=len(rows)//len(SEEDS), total=len(data))
    metrics = {}
    for a in ARMS:
        risk = [r["metrics"][a]["risk"] for r in rows if r["metrics"][a]["risk"]["selected"]]
        metrics[a] = dict(edits=sum(x["selected"] for x in risk), edited=len(risk), risk_pass=sum(x["supported"] for x in risk),
            easy_over_2=sum((x["known_easy_net_degradation"] or 0.) > .02 for x in risk),
            easy_undefined=sum(x["completion_easy_risk_upper"] is None for x in risk))
        metrics[a].update({k: av.interval(rows, lambda r: r["metrics"][a][k])
                           for k in ("ade", "fde", "gain", "hard", "tail")})
    if len({metrics[a]["edits"] for a in ARMS[2:-1]}) != 1:
        raise ValueError("Original matched budget changed")
    both = {k: {g: av.interval([r for r in rows if g == "all" or r["complete"]], lambda r: r["pair"][k])
                for g in ("all", "complete")} for k in rows[0]["pair"]}
    cal = {kind: {m: sum(x[kind][m] for x in frozen["folds"].values())
            for m in ("tested", "feasible", "local_strict", "strict_with_add", "accepted")}
           for kind in ("calibration", "diagnostic_add_cal")}
    action_fits = {r["action"]["checkpoint"]["sha256"] for r in frozen["folds"].values()}
    action_fits.update(a["checkpoint"]["sha256"] for r in frozen["folds"].values() for a in r["inner"].values())
    result = dict(registration=key, freeze=study.artifact(p), source="fresh_run",
        action_fits=len(action_fits), pair_fits=sum(r["pair"] is not None for r in frozen["folds"].values()),
        calibration=cal, counts=dict(counts), metrics=metrics, contrasts=both,
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False, deployment_changed=False)
    base.immutable(LOCAL/"rows.json", rows)
    base.immutable(OUT/"summary.json", result)
    print(study.json.dumps({k: result[k] for k in ("action_fits", "pair_fits", "calibration", "counts")}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "score"))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for d in (OUT, LOCAL):
        d.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, old, source, seal, parent = register()
        (train if phase == "fit" else evaluate)(key, old, source, seal, parent)
        beat(phase+"_complete")


if __name__ == "__main__":
    main()
