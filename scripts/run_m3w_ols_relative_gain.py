"""Matched CV- and OLS4-referenced training with an unchanged CV risk readout."""

import argparse
import fcntl
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_full_scene_harm_transfer as prior
from src.world_model import m3w_ols_relative_gain as study


cal, run, target, model, np, torch = prior.cal, prior.run, prior.target, prior.model, prior.np, prior.torch
PUBLIC = prior.PUBLIC.parent/"dronecrowd_ols_relative_gain_v1"
PRIVATE = prior.PRIVATE.parent/PUBLIC.name
STEPS, SAVE, BATCH, SEEDS = prior.STEPS, prior.SAVE, prior.BATCH, prior.SEEDS
ARMS = study.ARMS


def beat(phase, **fields):
    row = dict(pid=prior.os.getpid(), epoch=prior.time.time(), phase=phase, **fields)
    run.kin.prior.source.atomic(PRIVATE/"heartbeat.json", row)
    if phase == "train" or fields["completed"] % 5 == 0:
        print(row, flush=True)


def register():
    key, manifest, old, roster, floor, _ = prior.register()
    source = prior.PUBLIC.parent/"dronecrowd_observation_augmentation_v1"/"analysis.json"
    p = PUBLIC/"registration.json"
    run.once(p, run.stable(dict(prior_registration=key, motivation=run.artifact(source),
        code=[run.artifact(Path(x)) for x in (__file__, study.__file__)], arms=ARMS, seeds=SEEDS,
        steps=STEPS, batch=BATCH, checkpoint=SAVE, num_workers=0, torch_threads=4,
        training="Same native donor features, initialization, batches, Huber gain and mean-harm losses. Only signed gain anchor changes.",
        deployment="OLS4 default for all diagnostic arms. Compare gain heads at exact CV-gain OLS-replacement counts, and joint vs independent at exact joint counts.",
        joint="Search registered prices on predicted all/easy excess vs CV. Include unchanged OLS4 paths in every risk sum. No feasible forecast: retain OLS4 for diagnosis.",
        prices=study.PRICES, original_cv_2pct=True, original_sparse_protocol_unchanged=True,
        calibration_labels_for_training_or_tuning=False, confirmation_access=False, independent_confirmation=False)))
    return run.sha(p), manifest, old, roster, floor


def data():
    d = run.population()
    z = run.old.prior.arrays(("target_eval", "valid"))
    files = run.read(prior.DONORS)["files"]
    for k in ("target_eval", "valid"):
        run.checked(files[k])
    ids = d["global_ids"]
    if len(ids) != len(z["valid"]) or not np.array_equal(ids, np.sort(ids)):
        raise ValueError("Donor labels are not aligned")
    cut = run.read(cal.REG)["cut"]["easy"]
    groups = {}
    for q in d["groups"]:
        a, o, path, s = model.features(d["history"][q])
        y = z["target_eval"][q]-d["origin"][q, None]
        v = z["valid"][q]
        m, n, r = study.paired_targets(path, y, v, s, cut)
        groups.setdefault(str(d["sites"][q[0]]), []).append((a, o, m, n, r))
    out = {k: tuple(torch.from_numpy(np.concatenate(x)) for x in zip(*v, strict=True)) for k, v in sorted(groups.items())}
    if len(out) != 14 or sum(len(r[0]) for r in out.values()) != 125540:
        raise ValueError("Require complete screened donor population")
    return out


def fit(key, seed, donor, control):
    rows = list(donor.values())
    widths = [rows[0][j].shape[-1] for j in range(2)]
    ident = dict(registration=key, seed=seed, widths=widths, donors={k: len(r[0]) for k, r in donor.items()})
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed+1867)
    nets = torch.nn.ModuleList(model.HarmHead(*widths) for _ in range(2))
    nets[1].load_state_dict(nets[0].state_dict())
    opt = [torch.optim.AdamW(n.parameters(), lr=.001) for n in nets]
    path = PRIVATE/f"seed_{seed}.pt"
    start, trace = 0, []
    if path.exists():
        ck = torch.load(path, weights_only=True, map_location="cpu")
        if ck["identity"] != ident or ck["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Invalid paired checkpoint")
        start, trace = ck["step"], ck["trace"]
        nets.load_state_dict(ck["net"])
        for o, state in zip(opt, ck["optimizer"], strict=True):
            o.load_state_dict(state)
        rng.bit_generator.state = ck["rng"]
    for step in range(start+1, STEPS+1):
        counts = rng.multinomial(BATCH, np.full(len(rows), 1./len(rows)))
        picks = [(r, rng.integers(len(r[0]), size=n)) for r, n in zip(rows, counts, strict=True) if n]
        a, o, m, n, ref = [torch.cat([r[j][p] for r, p in picks]) for j in range(5)]
        losses = []
        for net, optimizer, y in zip(nets, opt, (m, n), strict=True):
            loss = model.loss(*net(a, o), y, ref)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        if step % SAVE == 0:
            trace.append(losses)
            tmp = path.with_suffix(".pending")
            torch.save(dict(identity=ident, step=step, net=nets.state_dict(), trace=trace,
                            optimizer=[o.state_dict() for o in opt], rng=rng.bit_generator.state), tmp)
            tmp.replace(path)
            beat("train", seed=seed, step=step, total=STEPS)
    ck = torch.load(run.checked(control["checkpoint"]), weights_only=True, map_location="cpu")
    if ck["identity"] != control["identity"] or ck["step"] != STEPS or ck["rng"] != rng.bit_generator.state:
        raise ValueError("Sampler not matched to original donor training")
    for name, val in nets[0].state_dict().items():
        torch.testing.assert_close(val, ck["net"][name], rtol=0., atol=0.)
    return dict(identity=ident, checkpoint=run.artifact(path), trace=trace, exact_cv_replay=True)


@torch.inference_mode()
def infer(src, d, coeff, nets):
    np.testing.assert_array_equal(src.query_frames, d["frames"])
    blocks, support = [], np.zeros(len(nets), dtype=int)
    for f, lo, hi in zip(d["frames"], d["offsets"][:-1], d["offsets"][1:], strict=True):
        x = src.get_inputs(int(f))
        at = x["history_complete"]
        h = x["positions"][at]
        np.testing.assert_array_equal(x["agent_ids"][at], d["agents"][lo:hi])
        np.testing.assert_array_equal(h[:, -1], d["origin"][lo:hi])
        a, o, paths, scale = model.features(h)
        np.testing.assert_array_equal(paths[:, 2:], model.floor.paths(coeff[lo:hi, :3]))
        paths[:, 0], paths[:, 1] = d["cv"][lo:hi], d["ols"][lo:hi]
        aa, oo = torch.from_numpy(a), torch.from_numpy(o)
        decisions = []
        for j, (cv, ols) in enumerate(nets):
            m, _ = cv(aa, oo)
            n, ref = ols(aa, oo)
            v, ok = study.decisions(paths, m.numpy()*scale[:, None, None],
                                    n.numpy()*scale[:, None, None], ref.numpy()*scale[:, None])
            decisions.append(v)
            support[j] += ok
        blocks.append(np.stack(decisions, axis=1))
    return np.concatenate(blocks), support.tolist()


def freeze(key, manifest, old, roster, floor):
    p = PUBLIC/"training.json"
    train = run.read(p)
    if train["registration"] != key or train["calibration_labels_used"] or tuple(v["identity"]["seed"] for v in train["fits"]) != SEEDS:
        raise ValueError("Wrong training roster")
    nets = []
    for v in train["fits"]:
        ck = torch.load(run.checked(v["checkpoint"]), weights_only=True, map_location="cpu")
        if ck["identity"] != v["identity"] or ck["step"] != STEPS or not v["exact_cv_replay"]:
            raise ValueError("Incomplete training")
        n = torch.nn.ModuleList(model.HarmHead(*ck["identity"]["widths"]) for _ in range(2))
        n.load_state_dict(ck["net"])
        nets.append(n.eval())
    meta, provenance, _ = target.context(roster)
    receipts = []
    for i, (cell, f) in enumerate(zip(old["clips"], floor["recordings"], strict=True), 1):
        rec = cell["sequence"]
        if f["recording"] != rec:
            raise ValueError("Wrong recording coefficient cache")
        mark = PRIVATE/f"{rec}.freeze.json"
        if not mark.exists():
            src = cal.reader(manifest, rec, meta, provenance)
            coeff = np.load(run.checked(f["coefficients"]), mmap_mode="r", allow_pickle=False)
            x, support = infer(src, target.old_arrays(cell), coeff, nets)
            choices = target.save_array(PRIVATE/f"{rec}.choices.npy", x)
            run.once(mark, run.stable(dict(registration=key, recording=rec, rows=len(x), choices=choices,
                                          coefficients=f["coefficients"], predicted_supported_queries=support)))
        row = run.read(mark)
        if row["registration"] != key or row["recording"] != rec or row["coefficients"] != f["coefficients"]:
            raise ValueError("Invalid frozen receipt")
        run.checked(row["choices"])
        receipts.append(row)
        beat("freeze", completed=i, total=len(roster))
    run.once(PUBLIC/"freeze.json", run.stable(dict(registration=key, records=receipts, training=run.artifact(p),
                                                  labels_used=False, confirmation_access=False)))


def score_recording(manifest, cell, f, meta, provenance, cut):
    rec = cell["sequence"]
    src = cal.reader(manifest, rec, meta, provenance)
    d = target.old_arrays(cell)
    y, v = cal.label_arrays(src, d)
    cv, ols, offsets = (d[k] for k in ("cv", "ols", "offsets"))
    idx = np.arange(len(cv))
    group = np.zeros(len(cv), dtype=int)
    w = np.repeat(1./np.diff(offsets), np.diff(offsets))
    x = np.load(run.checked(f["choices"]), mmap_mode="r", allow_pickle=False)
    coef = np.load(run.checked(f["coefficients"]), mmap_mode="r", allow_pickle=False)
    if x.shape != (len(cv), len(SEEDS), len(ARMS)) or x.max(initial=0) > 3:
        raise ValueError("Invalid frozen choice shape")
    bank = np.concatenate((ols[:, None], model.floor.paths(coef[:, :3])), axis=1)
    views = []
    for s in range(len(SEEDS)):
        out = {}
        for j, arm in enumerate(ARMS):
            path = bank[idx, x[:, s, j]]
            changed = (path != ols).any((1, 2))
            np.testing.assert_array_equal(changed, x[:, s, j] != 0)
            if arm in ("ols_matched", "ols_joint_matched"):
                expected = x[:, s, 0 if arm == "ols_matched" else 3] != 0
                np.testing.assert_array_equal(np.add.reduceat(changed.astype(int), offsets[:-1]),
                                              np.add.reduceat(expected.astype(int), offsets[:-1]))
            m = prior.policy(path, cv, cv, y, v, group, easy_cut=cut["easy"], hard_cut=cut["hard"])
            refs = {"cv": cv, "ols4": ols, "cv_gain": bank[idx, x[:, s, 0]], "ols_joint_matched": bank[idx, x[:, s, 4]]}
            out[arm] = dict(risk=m["recordings"]["0"], edits_over_ols=int(changed.sum()), ade=m["known_ADE_pixel"],
                fde=m["available_endpoint_FDE_diagnostic_pixel"], p95=m["p95_known_ADE_pixel"],
                **{k: prior.comparison_masses(r, path, cv, y, v, w) for k, r in refs.items()})
        views.append(out)
    return dict(views=views, rows=len(cv))


def summarize(rows, roster):
    groups = sorted(set(roster.values()))
    boot = np.random.default_rng(target.SEED).integers(len(groups), size=(target.DRAWS, len(groups)))
    compare = ("cv", "ols4", "cv_gain", "ols_joint_matched")
    by_group = {}
    for g in groups:
        views = [v for rec, r in rows.items() if roster[rec] == g for v in r["views"]]
        by_group[g] = {a: {k: prior.merge_comparisons([v[a][k] for v in views])["completion_lower_pp"]
                          for k in compare} for a in ARMS}
    metrics = {}
    for a in ARMS:
        x = [v[a] for r in rows.values() for v in r["views"]]
        risk = [v["risk"] for v in x]
        m = dict(edits_over_ols=sum(v["edits_over_ols"] for v in x), cv_changed=sum(r["selected"] for r in risk),
            risk_pass=sum(r["supported"] for r in risk), undefined_easy=sum(r["completion_easy_risk_upper"] is None for r in risk),
            easy_over_2=sum(r["completion_easy_risk_upper"] is not None and r["completion_easy_risk_upper"] > .02 for r in risk),
            dual_pass=sum(v["risk"]["supported"] and all(prior.merge_comparisons([v[k]])["completion_lower_pp"] > 0.
                          for k in ("cv", "ols4")) for v in x),
            ade=float(np.mean([v["ade"] for v in x if v["ade"] is not None])),
            fde=float(np.mean([v["fde"] for v in x if v["fde"] is not None])),
            worst_p95=max(v["p95"] for v in x if v["p95"] is not None))
        for k in compare:
            z = np.array([by_group[g][a][k] for g in groups])
            m[f"vs_{k}"] = dict(lower_pp=float(z.mean()), ci95=np.quantile(z[boot].mean(1), [.025, .975]).tolist(),
                                 positive_groups=int((z > 0.).sum()))
        metrics[a] = m
    return metrics, by_group


def score(key, manifest, old, roster, replay):
    p = PUBLIC/"freeze.json"
    freeze = run.read(p)
    if freeze["registration"] != key or freeze["labels_used"] or freeze["confirmation_access"]:
        raise ValueError("Scoring requires frozen outcome-free choices")
    run.checked(freeze["training"])
    meta, provenance, cut = target.context(roster)
    rows = {}
    for i, (cell, f) in enumerate(zip(old["clips"], freeze["records"], strict=True), 1):
        rec = cell["sequence"]
        if f["recording"] != rec:
            raise ValueError("Wrong frozen recording")
        path = PRIVATE/f"{rec}.score.json"
        if not path.exists():
            if replay:
                raise ValueError("Replay cannot read outcomes")
            out = score_recording(manifest, cell, f, meta, provenance, cut)
            run.once(path, run.stable(dict(registration=key, freeze=f, result=out)))
        r = run.read(path)
        if r["registration"] != key or r["freeze"] != f or r["result"]["rows"] != f["rows"]:
            raise ValueError("Wrong scoring receipt")
        rows[rec] = r["result"]
        beat("replay" if replay else "score", completed=i, total=len(roster))
    metrics, groups = summarize(rows, roster)
    out = dict(registration=key, freeze=run.artifact(p), metrics=metrics, groups=groups,
        rows=sum(r["rows"] for r in rows.values()), recordings=len(rows),
        predicted_supported_queries_by_seed=np.sum([r["predicted_supported_queries"] for r in freeze["records"]], axis=0).tolist(),
        result_source="fresh_run", original_cv_2pct=True, confirmation_access=False, deployment_changed=False)
    run.once(PUBLIC/"analysis.json", run.stable(out))
    print(metrics, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("train", "freeze", "score", "replay"))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for d in (PUBLIC, PRIVATE):
        d.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, manifest, old, roster, floor = register()
        if phase == "train":
            control = run.read(prior.PUBLIC/"training.json")
            d = data()
            fits = [fit(key, s, d, c) for s, c in zip(SEEDS, control["fits"], strict=True)]
            run.once(PUBLIC/"training.json", run.stable(dict(registration=key, fits=fits, calibration_labels_used=False)))
        elif phase == "freeze":
            freeze(key, manifest, old, roster, floor)
        else:
            score(key, manifest, old, roster, replay=phase == "replay")


if __name__ == "__main__":
    main()
