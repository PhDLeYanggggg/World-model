"""Build triple-excluded OpenDD forecasts for nested first-edit supervision."""

import argparse
import fcntl
from itertools import combinations
import os
from pathlib import Path
import platform
import sys
import time

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_slate_queue as study
from scripts import run_m3w_opendd_support_admission as admit

forecast, candidate, run, two = study.c.forecast, study.c, admit.run, admit.two
OUT = admit.OUT.parent/"opendd_nested_support_bank_v1"
PRIVATE = admit.PRIVATE.parent/OUT.name


def register():
    key, _, _, source, *_ = admit.register()
    paths = [admit.OUT/f"{k}.json" for k in ("registration", "analysis", "head_freeze")]
    r, a, h = map(run.read, paths)
    if (a["registration"] != key or h["registration"] != key or a["heads"] != run.sha(paths[2])
            or any(x["confirmation_access"] or not x["original_2pct"] for x in (r, a))):
        raise ValueError("Require frozen development-only support admission")
    _, upstream = forecast.author.verified_source()
    p = OUT/"registration.json"
    run.once(p, run.stable(dict(code=run.artifact(Path(__file__)),
        trainer=run.artifact(Path(two.fit.__file__)), forecaster=run.artifact(Path(forecast.model.__file__)),
        parents=[run.artifact(x) for x in paths], author=upstream, steps=forecast.STEPS,
        batch=forecast.BATCH, seeds=two.study.prior.SEEDS, threads=4, workers=0,
        native_steps=[8, 12], split="all unordered triples held out; fit on four other sites",
        architecture=forecast.model.ARCH, arm="comoving_ego", from_scratch=True,
        original_2pct=True, confirmation_access=False)))
    return run.sha(p), source


def save_prediction(key, triple, site, seed, p):
    saved = candidate.frozen_prediction
    return saved(key, triple, site, seed, p)


def execute(key, source, replay, beat):
    sites = {s: forecast.load(s, source) for s in admit.v.parent.source.SITES}
    heads, predictions, updates = {}, [], 0
    for triple in combinations(sorted(sites), 3):
        fit = [s for s in sites if s not in triple]
        sample = forecast.batcher([sites[s] for s in fit])
        for seed in two.study.prior.SEEDS:
            name = "producer_"+"_".join(triple)+f"_{seed}"
            ident = dict(registration=key, excluded=list(triple), fit=fit, seed=seed)
            net, heads[name], n = two.fit.fit(ident, name, seed, lambda: forecast.model.Forecast("comoving_ego"),
                                              forecast.STEPS, sample, forecast.model.loss, replay, beat)
            updates += n
            for site in triple:
                predictions.append(save_prediction(key, triple, site, seed, forecast.predict(net, sites[site][3])))
            beat(phase="producer", excluded=triple, seed=seed, updates=updates)
    run.once(OUT/"producer_freeze.json", run.stable(dict(registration=key, heads=heads,
        predictions=predictions, original_2pct=True, confirmation_access=False)))
    return updates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**p):
            z = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **p)
            run.kin.prior.source.atomic(PRIVATE/"heartbeat.json", z)
            if p.get("phase") != "fit":
                print(z, flush=True)

        two.fit.PRIVATE = candidate.PRIVATE = PRIVATE
        u = execute(*register(), a.replay, beat)
        beat(complete=True, updates=u, replay=a.replay)


if __name__ == "__main__":
    main()
