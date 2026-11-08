"""Test fixed-deadline coverage without refitting or threshold search."""

from collections import Counter
from pathlib import Path
import platform
import sys

import numpy as np

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_slate_queue as study
from scripts import run_m3w_opendd_dual_clock as clock
from src.world_model import m3w_opendd_dual_deadline as model

v, run, hard, two, prior = clock.v, clock.run, clock.hard, clock.two, clock.prior
OUT = clock.OUT
ARMS = ("deadline", "deadline_matched")


@torch.inference_mode()
def evaluate(d, bank, path, release, first, net, cuts):
    cv, ols = bank[:2]
    paths, features, blocks, gain, easy = clock.prepare(d, cv, path, release, first)
    y, valid, q = (d[k] for k in ("y", "valid", "query"))
    pred = {k: cv.copy() for k in ARMS}
    count = Counter()
    for seq in blocks.values():
        packets = clock.nominate(seq, gain, easy)

        def score(t, p, state, menu):
            x = np.stack([clock.model.describe(features, t, p, state, u) for u in menu])
            return net(torch.from_numpy(x)).numpy()

        plan = model.stream(iter(packets), score)
        roles = {k for k, _ in plan}
        control = clock.model.earliest(packets, roles)
        if len(plan) != len(roles) or {k for k, _ in control} != roles:
            raise ValueError("Deadline policy did not match its role budget")
        for arm, selected in zip(ARMS, (plan, control), strict=True):
            if len({i for _, i in selected}) != len(selected):
                raise ValueError("Deadline reused a recipient")
            for r, i in selected:
                pred[arm][i] = paths[r][i]
        matched = dict(control)
        count.update(gain=int(0 in roles), support=int(1 in roles), both=int(len(roles) == 2),
                     retimed=int(sum(q[i] != q[matched[k]] for k, i in plan)))
    stats = {k: v.parent.score(d, cv, ols, p, cuts) for k, p in pred.items()}
    if any(s["edits"] != count["gain"]+count["support"] for s in stats.values()):
        raise ValueError("Deadline changed scored edit counts")
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    pairs = [("deadline", "deadline_matched"), *((k, "cv") for k in ARMS)]
    effects = {f"{a}_vs_{b}": hard.paired_pp(cv if b == "cv" else pred[b], pred[a], y, valid, mass)
               for a, b in pairs}
    return dict(metrics=stats, effects=effects, diagnostics=count)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, outer, old, source, gains, _, _, _, _ = clock.register()
    p, h, f = (OUT/f"{k}.json" for k in ("analysis", "head_freeze", "serialization_recovery_v2"))
    a, b, c = map(run.read, (p, h, f))
    if (a["registration"] != key or a["heads"] != run.sha(h) or b["registration"] != key
            or c["heads"]["sha256"] != run.sha(h)
            or any(z["confirmation_access"] or not z["original_2pct"] for z in (a, c))):
        raise ValueError("Require verified recovered dual-clock checkpoints")
    r = OUT/"deadline_registration.json"
    run.once(r, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parents=[run.artifact(z) for z in (p, h, f)], original_2pct=True, confirmation_access=False,
        no_new_training=True, post_result_development_diagnostic=True,
        deadline=clock.model.LOOKAHEAD, terminal_forcing=False,
        matching="same per-record realized roles; retrospective earliest nomination; not radius matched",
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"))))
    sites = {s: study.c.forecast.load(s, source)[0] for s in v.parent.source.SITES}
    result = {}
    for s, d in sites.items():
        bank = v.parent.prior.analytic(d)
        cut = np.quantile(np.concatenate([z["cut_samples"] for t, z in sites.items() if t != s]), [.25, .75])
        width = hard.model.features(d["h"][:1]).shape[1]
        result[s] = []
        for row in old["by_site"][s]:
            seed = row["seed"]
            name = f"{s}_{seed}"
            release = two.fit.load(outer[name], lambda: hard.model.network(width), hard.STEPS)
            with np.load(run.checked(row["prediction"]), allow_pickle=False) as z:
                path = z["comoving_ego"]
            n = prior.feature({k: d[k][:1] for k in ("h", "query")}, bank[0][:1], path[:1])[0].shape[1]
            head = two.fit.load(gains[name], lambda: prior.model.network(n), prior.STEPS)
            net = two.fit.load(b["heads"][name], lambda: clock.model.network(4*(n+4)+3), clock.STEPS)
            result[s].append(dict(seed=seed, **evaluate(d, bank, path, release, head, net, cut)))
        print(s, "fixed deadline evaluated", flush=True)
    flat = [z for group in result.values() for z in group]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(k):
        x = np.array([np.mean([z["effects"][k] for z in group]) for group in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(x.min()))

    summary = dict(metrics={k: {f: sum(z["metrics"][k][f] for z in flat) for f in v.prior.FIELDS} for k in ARMS},
        effects={k: ci(k) for k in flat[0]["effects"]},
        diagnostics=sum((z["diagnostics"] for z in flat), Counter()))
    run.once(OUT/"deadline_analysis.json", run.stable(dict(registration=run.sha(r), by_site=result,
        summary=summary, original_2pct=True, confirmation_access=False, submission_ready=False)))
    print(run.stable(summary), flush=True)


if __name__ == "__main__":
    main()
