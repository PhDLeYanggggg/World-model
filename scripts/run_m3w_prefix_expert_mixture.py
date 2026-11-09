"""Fit matched causal mixture predictors on the original moving-source split."""

import argparse
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == "Darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 .venv-pytorch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_kinematic_forecast as prior
from scripts import run_m3w_recording_count as trainer
from src.evaluation import m3w_grouped_recording_readout as grouped
from src.world_model import m3w_prefix_expert_mixture as model

np, torch, read, artifact, checked, sha, once, stable = (
    prior.np, prior.torch, prior.read, prior.artifact, prior.checked, prior.sha, prior.once, prior.stable)
OUT = prior.PUBLIC.parent / "moving_prefix_expert_mixture_v1"
PRIVATE = prior.PRIVATE.parent / OUT.name
STEPS, BATCH, SEEDS = 1000, 128, (17, 29, 43)
CV, OLS = prior.parent.CV, prior.parent.OLS
PAIRS = {f"{a}_vs_{k}": (k, a, CV) for a in model.ARMS for k in
         (CV, OLS, "ols8", "eqmotion", "previous_kinematic")}
PAIRS.update(prefix_vs_kinematic=("kinematic", "prefix", CV), prefix_vs_static=("static", "prefix", CV))


def register():
    jobs = prior.register()
    original = prior.PUBLIC / "analysis.json"
    verification = prior.PUBLIC / "replay_execution.json"
    if read(verification)["analysis"]["sha256"] != sha(original):
        raise ValueError("Require verified previous kinematic score")
    refs = [prior.REG, prior.PUBLIC / "forecast_freeze.json", original, verification,
            prior.prior.PUBLIC / "fit_population.json", prior.parent.REG]
    reg = OUT / "registration.json"
    once(reg, stable(dict(files=[artifact(p) for p in (Path(__file__), Path(model.__file__),
        Path(trainer.__file__))], parents=[artifact(p) for p in refs],
        arms=model.ARMS, experts=model.EXPERTS, steps=STEPS, batch=BATCH, seeds=SEEDS,
        inputs="eight positions only; prefix replay from observed prefixes 4..7 into observed suffix",
        control="identical network, draws, initialization and step budget; zero feature blocks",
        fit="same nine original fit groups; observed-step distance normalized by past speed",
        eval="all 34 exposed A recordings/14 groups, no tuning or horizon change",
        primary="paired full-grid gain versus OLS4, CV and matched kinematic control",
        original_2pct=True, confirmation_access=False, train_only_selection=True)))
    return sha(reg), jobs


def fitting(jobs, beat):
    path = PRIVATE / "fit_arrays.npz"
    receipt = PRIVATE / "fit_arrays.json"
    parent = prior.prior.PUBLIC / "fit_population.json"
    identity = dict(parent=artifact(parent), ids=jobs[0]["identity"]["train_ids_sha256"])
    if not path.exists():
        d = prior.prior.fitting_only(jobs[0]["identity"], beat)
        tmp = path.with_suffix(".pending.npz")
        np.savez(tmp, history=d["geometry"][:, :16].reshape(-1, 8, 2),
                 target=d["target"], valid=d["valid"], ids=d["ids"], sites=d["sites"])
        tmp.replace(path)
        once(receipt, stable(dict(identity=identity, arrays=artifact(path))))
    binding = read(receipt)
    if binding["identity"] != identity or checked(binding["arrays"]) != path:
        raise ValueError("Fitting inputs changed")
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    if len(d["ids"]) != 69746 or prior.prior.source.array_hash(d["ids"]) != identity["ids"]:
        raise ValueError("Fitting roster differs")
    return d


def tensors(h, arm, candidates):
    x, scale = model.features(h, arm)
    return torch.from_numpy(x), candidates, torch.from_numpy(scale)


def sample_fn(data, group):
    def sample(rng):
        sizes = rng.multinomial(BATCH, np.full(len(group), 1. / len(group)))
        ids = np.concatenate([g[rng.integers(len(g), size=n)] for g, n in zip(group, sizes, strict=True)])
        return tuple(v[ids] for v in data)
    return sample


@torch.inference_mode()
def predict(net, x, candidates):
    rows, weights = [], []
    for a, b in zip(x.split(2048), candidates.split(2048), strict=True):
        p, w = net(a, b)
        rows.append(p.numpy())
        weights.append(w.numpy())
    return np.concatenate(rows), np.concatenate(weights).mean(0).astype(float).tolist()


def freeze(name, ids, pred):
    f = PRIVATE / f"{name}.npz"
    if f.exists():
        with np.load(f, allow_pickle=False) as z:
            np.testing.assert_array_equal(z["ids"], ids)
            np.testing.assert_array_equal(z["pred"], pred)
    else:
        tmp = f.with_suffix(".pending.npz")
        np.savez(tmp, ids=ids, pred=pred)
        tmp.replace(f)
    return artifact(f)


def fit_predict(reg, jobs, replay, beat):
    d = fitting(jobs, beat)
    arrays = prior.bank.old.prior.arrays(("global_ids", "geometry", "sites"))
    ids, site = arrays["global_ids"], arrays["sites"]
    if len(ids) != 125540 or set(d["sites"]) & set(site):
        raise ValueError("Require all exposed A sites distinct from fitting")
    train_h, test_h = d["history"], arrays["geometry"][:, :16].reshape(-1, 8, 2)
    c = torch.from_numpy(model.bank(train_h).astype(np.float32))
    tc = torch.from_numpy(model.bank(test_h).astype(np.float32))
    labels = tuple(torch.from_numpy(d[k]) for k in ("target", "valid"))
    groups = [np.flatnonzero(d["sites"] == s) for s in sorted(set(d["sites"]))]
    cells, updates = [], 0
    for arm in model.ARMS:
        x = tensors(train_h, arm, c)
        tx = tensors(test_h, arm, tc)[0]
        sample = sample_fn((*x, *labels), groups)
        for seed in SEEDS:
            name = f"{arm}_{seed}"
            ident = dict(registration=reg, seed=seed, arm=arm, fit_ids=jobs[0]["identity"]["train_ids_sha256"])
            net, meta, u = trainer.fit(ident, name, seed, lambda: model.Mixture(x[0].shape[1]),
                                       STEPS, sample, model.loss, replay, beat)
            updates += u
            pred, w = predict(net, tx, tc)
            cells.append(dict(seed=seed, arm=arm, forecast=freeze(name, ids, pred), mean_weights=w, fit=meta))
            beat(state="forecast_frozen", cell=name, completed=len(cells), updates=updates)
    f = OUT / "freeze.json"
    once(f, stable(dict(registration=reg, before_A_labels=True, cells=cells)))
    return f, updates


def score(reg, f):
    frozen = read(f)
    if len(frozen["cells"]) != 9 or not frozen["before_A_labels"] or frozen["registration"] != reg:
        raise ValueError("All forecasts must be frozen before scoring")
    old, parent = prior.bank.old, prior.parent
    a = old.prior.arrays(("global_ids", "history", "origin", "queries", "sites", "recordings"))
    labels = old.prior.arrays(("target_eval", "valid"))
    ids, q, site, rec = (a[k] for k in ("global_ids", "queries", "sites", "recordings"))
    cuts = read(old.PARENTS[4])["cuts"]
    cut = {s: cuts[str(x["job"]["producer_fold"])] for x in read(old.PARENTS[3])["forecasts"]
           for s in x["job"]["donor_sites"]}
    h = a["history"].astype(float)
    b = model.bank(h)
    truth, valid = labels["target_eval"] - a["origin"][:, None], labels["valid"]
    roster, rows = {}, []
    previous = read(prior.PUBLIC / "forecast_freeze.json")["forecasts"]
    for seed in SEEDS:
        bank = parent.read_banks(read(parent.REG)["forecasts"], seed, ids, site)
        paths = {"eqmotion": bank["eqmotion"], CV: b[:, 0], OLS: b[:, 1], "ols8": b[:, 2]}
        for k, p in zip(model.EXPERTS[3:], b[:, 3:].swapaxes(0, 1), strict=True):
            paths[k] = p
        for cell in frozen["cells"]:
            if cell["seed"] == seed:
                with np.load(checked(cell["forecast"]), allow_pickle=False) as z:
                    np.testing.assert_array_equal(z["ids"], ids)
                    paths[cell["arm"]] = z["pred"].astype(float)
        p, = [v for v in previous if v["seed"] == seed and v["arm"] == "self"]
        with np.load(checked(p["forecast"]), allow_pickle=False) as z:
            np.testing.assert_array_equal(z["ids"], ids)
            paths["previous_kinematic"] = z["prediction"].astype(float)
        for rec0 in np.unique(rec):
            ix = np.flatnonzero(rec == rec0)
            key, s = str(rec0), str(site[ix[0]])
            if not np.all(site[ix] == s):
                raise ValueError("Recording crosses source groups")
            roster[key] = s
            local = {k: v[ix] for k, v in paths.items()}
            r = parent.readout.score_recording(local, {}, truth[ix], valid[ix], q[ix], key,
                                               easy_cut=cut[s]["easy"], hard_cut=cut[s]["hard"])
            _, inv, n = np.unique(q[ix], return_inverse=True, return_counts=True)
            w = 1. / n[inv]
            r["comparisons"] = {k: model.full_masses(local[l], local[p], local[d], truth[ix], valid[ix], w)
                                for k, (l, p, d) in PAIRS.items()}
            rows.append(dict(r, source=s, seed=seed))
    result = grouped.grouped_readout(rows, roster, SEEDS)
    if result["views"] != 102 or result["source_groups"] != 14:
        raise ValueError("Full source roster required")
    once(OUT / "analysis.json", stable(dict(registration=reg, freeze=sha(f), summary=result, rows=rows,
        fit_rows=69746, exposed_rows=len(ids), original_2pct=True, confirmation_access=False,
        deployment_changed=False, submission_ready=False)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    args = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    trainer.PRIVATE = PRIVATE
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            state = dict(pid=os.getpid(), seconds=round(time.monotonic() - start, 3), **v)
            prior.prior.source.atomic(PRIVATE / "heartbeat.json", state)
            if v.get("step", STEPS) in (100, STEPS):
                print(state, flush=True)

        reg, jobs = register()
        f, updates = fit_predict(reg, jobs, args.replay, beat)
        score(reg, f)
        beat(state="complete", updates=updates, replay=args.replay)


if __name__ == "__main__":
    main()
