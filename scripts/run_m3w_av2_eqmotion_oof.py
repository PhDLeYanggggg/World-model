"""Build city-excluded EqMotion candidates for AV2 selector supervision."""

from collections import defaultdict
import fcntl
import os
from pathlib import Path
import platform
import sys
from time import time

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_eqmotion as eq

av, prev, np, torch, study, base = eq.av, eq.prev, eq.np, eq.torch, eq.study, eq.base
OUT, LOCAL = eq.OUT.parent/"av2_eqmotion_oof_v1", eq.LOCAL.parent/"av2_eqmotion_oof_v1"


def beat(phase, **kw):
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(dict(phase=phase, pid=os.getpid(), epoch=time(), **kw))+"\n")
    p.replace(LOCAL/"heartbeat.json")


def layout(cities):
    roles, jobs = {}, defaultdict(set)
    for held in sorted(cities):
        cal, donors = prev.roles(cities, held)
        inner = {}
        for city in donors:
            group = tuple(s for s in donors if s != city)
            jobs[group].add(city)
            inner[city] = "+".join(group)
        roles[held] = dict(calibration=cal, donors=donors, inner=inner)
    return roles, {"+".join(k): (k, sorted(v)) for k, v in sorted(jobs.items())}


def register():
    parent, previous, source, _, costs, relatives = eq.register()
    path = eq.OUT/"summary.json"
    report = study.read(path)
    freeze = study.read(study.checked(report["freeze"]))
    if (report["registration"] != parent or freeze["registration"] != parent or freeze["held_labels_used"]
            or freeze["calibration_labels_used"] or report["reserved_outcomes_opened"]
            or report["independent_confirmation"]):
        raise ValueError("Require the original frozen EqMotion control")
    data = eq.scale.read_inputs(previous, source)
    roles, jobs = layout({d["city"] for d in data.values()})
    reg = OUT/"registration.json"
    base.immutable(reg, dict(parent=study.artifact(path), parent_freeze=report["freeze"],
        code=[study.artifact(Path(x)) for x in (__file__, eq.__file__, eq.eq.__file__)],
        roles=roles, jobs=jobs, seeds=eq.SEEDS, steps=eq.STEPS, batch=eq.BATCH, checkpoint=eq.SAVE,
        training="exact same K=1 centered EqMotion recipe; exclude each predicted donor plus both outer excluded cities",
        inference="only past context; fixed parent EqMotion for held inference",
        selection="no new candidate families, selection or risk-rule changes at this phase",
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(reg), data, source, roles, jobs, freeze, costs, relatives


def fit(key, label, donors, seed, pools, completed, total):
    name = f"{label}_{seed}"
    path = LOCAL/f"{name}.pt"
    ident = dict(registration=key, donors=list(donors), seed=seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed+3711)
    net = eq.eq.Forecast(eq.VARIANT)
    opt = torch.optim.AdamW((p for p in net.parameters() if p.requires_grad), lr=.001)
    step, trace = 0, []
    if path.exists():
        ck = torch.load(path, weights_only=True, map_location="cpu")
        if ck["identity"] != ident or ck["step"] not in range(eq.SAVE, eq.STEPS+1, eq.SAVE):
            raise ValueError("Incorrect nested producer checkpoint")
        step, trace = ck["step"], ck["loss"]
        rng.bit_generator.state = ck["rng"]
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
    for i in range(step+1, eq.STEPS+1):
        sizes = rng.multinomial(eq.BATCH, np.full(len(pools), 1./len(pools)))
        idx = [rng.integers(len(d[0]), size=n) for d, n in zip(pools, sizes, strict=True)]
        data = [torch.cat([d[k][ix] for d, ix in zip(pools, idx, strict=True)]) for k in range(6)]
        loss = eq.eq.loss(net, data[:4], data[4], data[5])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if i % eq.SAVE == 0:
            trace.append(float(loss.detach()))
            pending = path.with_suffix(".pending")
            torch.save(dict(identity=ident, step=i, net=net.state_dict(), opt=opt.state_dict(),
                            rng=rng.bit_generator.state, loss=trace), pending)
            pending.replace(path)
            beat("fit", model=name, step=i, total_steps=eq.STEPS, completed=completed, total=total)
    return net.eval(), dict(identity=ident, checkpoint=study.artifact(path), loss=trace,
        donor_actors=sum(len(d[0]) for d in pools))


def generate(key, data, source, roles, jobs):
    receipts = {}
    for label, (donors, cities) in jobs.items():
        if set(donors) & set(cities):
            raise ValueError("Nested fit sees a predicted city")
        names = [n for n, d in data.items() if d["city"] in cities]
        paths = [LOCAL/f"{label}_{s}.json" for s in eq.SEEDS]
        if not all(p.exists() for p in paths):
            y = prev.load_targets(source, data, [n for n, d in data.items() if d["city"] in donors])
            pools = eq.city_batches(data, y, donors)
        for seed, receipt in zip(eq.SEEDS, paths, strict=True):
            name = f"{label}_{seed}"
            if not receipt.exists():
                net, item = fit(key, label, donors, seed, pools, len(receipts), len(jobs)*len(eq.SEEDS))
                forecast, offsets = eq.predict(net, data, names)
                file = av.turn.exp.previous.save_bank(LOCAL/f"{name}.npz", dict(
                    registration=key, names=np.asarray(names), offsets=offsets, forecast=forecast))
                base.immutable(receipt, dict(fit=item, forecast=file, predicted_cities=cities, scenes=len(names)))
            r = study.read(receipt)
            if (r["fit"]["identity"] != dict(registration=key, donors=list(donors), seed=seed)
                    or r["predicted_cities"] != cities or r["scenes"] != len(names)):
                raise ValueError("Nested checkpoint receipt mismatch")
            study.checked(r["fit"]["checkpoint"])
            study.checked(r["forecast"])
            receipts[name] = r
            beat("freeze", model=name, completed=len(receipts), total=len(jobs)*len(eq.SEEDS))
    for held, role in roles.items():
        for city, producer in role["inner"].items():
            for seed in eq.SEEDS:
                r = receipts[f"{producer}_{seed}"]
                train = set(r["fit"]["identity"]["donors"])
                if train != set(role["donors"])-{city} or train & {held, role["calibration"], city}:
                    raise ValueError("Improper out-of-fold neural candidate")
    out = dict(registration=key, models=receipts, roles=roles, past_only_forecasts=True,
        independent_confirmation=False, reserved_outcomes_opened=False)
    base.immutable(OUT/"freeze.json", out)
    beat("complete", completed=len(receipts), total=len(jobs)*len(eq.SEEDS))


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for d in (OUT, LOCAL):
        d.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, data, source, roles, jobs, _, _, _ = register()
        generate(key, data, source, roles, jobs)


if __name__ == "__main__":
    main()
