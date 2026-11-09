"""Test donor-only observation augmentation at the original learned gain coverage."""

import argparse
import fcntl
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_full_scene_harm_transfer as plain
from src.world_model import m3w_observation_augmentation as aug


np, torch, run, cal, target, model = plain.np, plain.torch, plain.run, plain.cal, plain.target, plain.model
PUBLIC, PRIVATE = plain.PUBLIC.parent/"dronecrowd_observation_augmentation_v1", plain.PRIVATE.parent/"dronecrowd_observation_augmentation_v1"
MODES = ("scale", "noise")
ARMS = ("plain_gain", "scale_matched", "noise_matched", "scale_gain", "noise_gain", "noise_scene")
STEPS, SAVE, BATCH, SEEDS = plain.STEPS, plain.SAVE, plain.BATCH, plain.SEEDS


def beat(phase, **fields):
    v = dict(pid=plain.os.getpid(), epoch=plain.time.time(), phase=phase, **fields)
    run.kin.prior.source.atomic(PRIVATE/"heartbeat.json", v)
    if fields.get("completed", 0) % 5 == 0:
        print(v, flush=True)


def register():
    key, manifest, old, roster, parent, _ = plain.register()
    source = plain.PUBLIC/"analysis.json"
    r = run.read(source)
    seal = run.read(run.checked(r["freeze"]))
    training = run.read(run.checked(seal["training"]))
    if (key != r["registration"] or key != seal["registration"] or seal["held_labels_used"]
            or r["confirmation_access"] or training["calibration_labels_used"]
            or len(parent["recordings"]) != len(seal["records"])):
        raise ValueError("Require the completed and frozen unaugmented comparison")
    p = PUBLIC/"registration.json"
    run.once(p, run.stable(dict(code=[run.artifact(Path(p)) for p in (__file__, aug.__file__)],
        source=run.artifact(source), prior_training=seal["training"], prior_freeze=r["freeze"],
        augmentation="half of donor queries unchanged; otherwise factor 2**Uniform(-6,0) in shared query coordinates; "
                     "same factor on history and clean future, add iid .5 px Gaussian noise only to observed history in noise arm; "
                     "rebase labels to perturbed observed origin, preserve all masks",
        augmentation_randomness="independent seed stream, same donor queries and factors in scale and noise fits; no target statistics used",
        architecture_and_sampler="same as prior fit, same initial weights, same site and row draws and update count",
        arms=ARMS, primary="noise_matched minus plain_gain, exact original count per query",
        scope="full-scene donor-only augmentation diagnostic; no new target roles, inference perturbation, or threshold tuning",
        steps=STEPS, checkpoint=SAVE, batch=BATCH, seeds=SEEDS, original_2pct=True, torch_threads=4,
        num_workers=0, confirmation_access=False, independent_confirmation=False)))
    return run.sha(p), manifest, old, roster, parent, seal, r, training


def donor_batches(d, labels, seed):
    rng = np.random.default_rng(seed+27053)
    data = {}
    cut = run.read(cal.REG)["cut"]["easy"]
    for q in d["groups"]:
        site = str(d["sites"][q[0]])
        h, y, v = d["history"][q], labels["target_eval"][q], labels["valid"][q]
        apply = rng.random() >= aug.KEEP_NATIVE
        factor = 2.**rng.uniform(aug.MIN_POWER, 0.) if apply else 1.
        jitter = rng.normal(0., aug.NOISE_SD, size=h.shape) if apply else None
        row = data.setdefault(site, {k: [] for k in MODES})
        for mode in MODES:
            if apply:
                x, future = aug.transform(h, y, v, factor, jitter if mode == "noise" else None)
            else:
                x, future = h, y-h[:, -1:]
            a, o, paths, s = model.features(x)
            m, ref = model.targets(paths, future, v, s, cut)
            row[mode].append((a, o, m, ref))
    return {k: {m: tuple(torch.from_numpy(np.concatenate(a)) for a in zip(*v[m], strict=True))
                for m in MODES} for k, v in sorted(data.items())}


def fit(key, seed, rows, parent):
    donors = {k: len(v["scale"][0]) for k, v in rows.items()}
    if donors != parent["identity"]["donors"]:
        raise ValueError("Augmented donors differ from the original fit")
    widths = parent["identity"]["widths"]
    ident = dict(registration=key, seed=seed, donors=donors, widths=widths)
    rng = np.random.default_rng(seed+1867)
    torch.manual_seed(seed)
    net = torch.nn.ModuleDict({k: model.HarmHead(*widths) for k in MODES})
    net["noise"].load_state_dict(net["scale"].state_dict())
    opt = {k: torch.optim.AdamW(net[k].parameters(), lr=.001) for k in MODES}
    path = PRIVATE/f"seed_{seed}.pt"
    start, trace = 0, []
    if path.exists():
        ck = torch.load(path, weights_only=True, map_location="cpu")
        if ck["identity"] != ident or ck["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Incorrect augmentation checkpoint")
        start, trace = ck["step"], ck["trace"]
        net.load_state_dict(ck["net"])
        for k in MODES:
            opt[k].load_state_dict(ck["optimizer"][k])
        rng.bit_generator.state = ck["rng"]
    groups = list(rows.values())
    for step in range(start+1, STEPS+1):
        n = rng.multinomial(BATCH, np.full(len(groups), 1./len(groups)))
        picks = [(r, rng.integers(len(r["scale"][0]), size=int(count))) for r, count in zip(groups, n, strict=True) if count]
        losses = []
        for k in MODES:
            a, o, y, ref = [torch.cat([r[k][i][j] for r, j in picks]) for i in range(4)]
            loss = model.loss(*net[k](a, o), y, ref)
            opt[k].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net[k].parameters(), 5., error_if_nonfinite=True)
            opt[k].step()
            losses.append(float(loss.detach()))
        if step % SAVE == 0:
            trace.append(losses)
            tmp = path.with_suffix(".pending")
            torch.save(dict(identity=ident, step=step, trace=trace, net=net.state_dict(),
                            optimizer={k: o.state_dict() for k, o in opt.items()}, rng=rng.bit_generator.state), tmp)
            tmp.replace(path)
            beat("train", seed=seed, step=step, total=STEPS)
    ck = torch.load(run.checked(parent["checkpoint"]), weights_only=True, map_location="cpu")
    if ck["identity"] != parent["identity"] or ck["rng"] != rng.bit_generator.state or ck["step"] != STEPS:
        raise ValueError("Training draws do not match the frozen control")
    return dict(identity=ident, checkpoint=run.artifact(path), trace=trace, same_final_sampler=True)


def train(key, parent):
    d = run.population()
    labels = run.old.prior.arrays(("target_eval", "valid"))
    if len(labels["valid"]) != len(d["global_ids"]) or not np.array_equal(d["global_ids"], np.sort(d["global_ids"])):
        raise ValueError("Donor inputs and targets disagree")
    fit_rows = [fit(key, s, donor_batches(d, labels, s), ref) for s, ref in zip(SEEDS, parent["fits"], strict=True)]
    run.once(PUBLIC/"training.json", run.stable(dict(registration=key, fits=fit_rows,
        donor_rows=len(d["global_ids"]), calibration_labels_used=False)))


def load_nets(key):
    r = run.read(PUBLIC/"training.json")
    if r["registration"] != key or r["calibration_labels_used"] or tuple(f["identity"]["seed"] for f in r["fits"]) != SEEDS:
        raise ValueError("Invalid augmented checkpoint roster")
    out = []
    for fit_ref in r["fits"]:
        ck = torch.load(run.checked(fit_ref["checkpoint"]), weights_only=True, map_location="cpu")
        if ck["identity"] != fit_ref["identity"] or ck["step"] != STEPS or not fit_ref["same_final_sampler"]:
            raise ValueError("Incomplete or unmatched augmented weights")
        net = torch.nn.ModuleDict({k: model.HarmHead(*ck["identity"]["widths"]) for k in MODES})
        net.load_state_dict(ck["net"])
        out.append(net.eval())
    return out


@torch.inference_mode()
def decisions(reader, prev, coeff, prior, nets):
    np.testing.assert_array_equal(reader.query_frames, prev["frames"])
    out = []
    for frame, lo, hi in zip(prev["frames"], prev["offsets"][:-1], prev["offsets"][1:], strict=True):
        x = reader.get_inputs(int(frame))
        at = x["history_complete"]
        h = x["positions"][at]
        np.testing.assert_array_equal(x["agent_ids"][at], prev["agents"][lo:hi])
        np.testing.assert_array_equal(h[:, -1], prev["origin"][lo:hi])
        a, o, paths, scale = model.features(h)
        np.testing.assert_array_equal(paths[:, 2:], model.floor.paths(coeff[lo:hi, :3]))
        cv, ols = prev["cv"][lo:hi], prev["ols"][lo:hi]
        legal = (np.concatenate((ols[:, None], paths[:, 2:]), axis=1) != cv[:, None]).any((2, 3))
        aa, oo = torch.from_numpy(a), torch.from_numpy(o)
        rows = []
        for j, net in enumerate(nets):
            p = prior[lo:hi, j, plain.ARMS.index("gain")]
            count = int((p != 0).sum())
            chosen, matched = {}, {}
            for k in MODES:
                m, ref = net[k](aa, oo)
                m, ref = m.numpy()*scale[:, None, None], ref.numpy()*scale[:, None]
                chosen[k] = model.choose(m, ref, legal)
                matched[k] = aug.match_gain(m[..., 0], legal, count)
            rows.append(np.column_stack((p, matched["scale"], matched["noise"],
                                         chosen["scale"][:, 0], chosen["noise"][:, 0], chosen["noise"][:, 2])))
        out.append(np.stack(rows, axis=1))
    return np.concatenate(out)


def freeze(key, manifest, old, roster, parent, earlier):
    meta, provenance, _ = target.context(roster)
    nets = load_nets(key)
    all_rows = []
    for i, (cell, base, before) in enumerate(zip(old["clips"], parent["recordings"], earlier["records"], strict=True), 1):
        rec = cell["sequence"]
        if base["recording"] != rec or before["recording"] != rec:
            raise ValueError("Wrong frozen calibration source")
        mark = PRIVATE/f"{rec}.freeze.json"
        if not mark.exists():
            src = cal.reader(manifest, rec, meta, provenance)
            coeff = np.load(run.checked(base["coefficients"]), allow_pickle=False, mmap_mode="r")
            prior = np.load(run.checked(before["choices"]), allow_pickle=False, mmap_mode="r")
            at = decisions(src, target.old_arrays(cell), coeff, prior, nets)
            p = target.save_array(PRIVATE/f"{rec}.choices.npy", at)
            run.once(mark, run.stable(dict(registration=key, recording=rec, rows=len(at), choices=p,
                                           source_coefficients=base["coefficients"], source_choices=before["choices"])))
        r = run.read(mark)
        if (r["registration"] != key or r["recording"] != rec or r["source_coefficients"] != base["coefficients"]
                or r["source_choices"] != before["choices"]):
            raise ValueError("Wrong augmentation inference checkpoint")
        run.checked(r["choices"])
        all_rows.append(r)
        beat("freeze", completed=i, total=len(roster))
    run.once(PUBLIC/"freeze.json", run.stable(dict(registration=key, records=all_rows, training=run.artifact(PUBLIC/"training.json"),
                                                  labels_used=False, confirmation_access=False)))


def score_recording(manifest, cell, f, meta, provenance, cut):
    rec = cell["sequence"]
    src = cal.reader(manifest, rec, meta, provenance)
    old = target.old_arrays(cell)
    y, v = cal.label_arrays(src, old)
    c, o, offsets = (old[k] for k in ("cv", "ols", "offsets"))
    index, group = np.arange(len(c)), np.zeros(len(c), dtype=int)
    w = np.repeat(1./np.diff(offsets), np.diff(offsets))
    coeff = np.load(run.checked(f["source_coefficients"]), allow_pickle=False, mmap_mode="r")
    at = np.load(run.checked(f["choices"]), allow_pickle=False, mmap_mode="r")
    if at.shape != (len(c), len(SEEDS), len(ARMS)) or at.max(initial=0) > 4:
        raise ValueError("Frozen decisions don't match roster")
    bank = np.concatenate((c[:, None], o[:, None], model.floor.paths(coeff[:, :3])), axis=1)
    seeds = []
    for s in range(len(SEEDS)):
        prior = bank[index, at[:, s, 0]]
        expected = np.add.reduceat((prior != c).any((1, 2)).astype(int), offsets[:-1])
        view = {}
        for j, k in enumerate(ARMS):
            p = bank[index, at[:, s, j]] if j else prior
            changed = (p != c).any((1, 2))
            np.testing.assert_array_equal(changed, at[:, s, j] != 0)
            if k.endswith("_matched"):
                np.testing.assert_array_equal(np.add.reduceat(changed.astype(int), offsets[:-1]), expected)
            m = plain.policy(p, c, c, y, v, group, easy_cut=cut["easy"], hard_cut=cut["hard"])
            view[k] = dict(risk=m["recordings"]["0"], ade=m["known_ADE_pixel"], fde=m["available_endpoint_FDE_diagnostic_pixel"],
                p95=m["p95_known_ADE_pixel"], hard=m["subsets"]["hard"]["gain_percent"],
                **{b: plain.comparison_masses(r, p, c, y, v, w) for b, r in (("cv", c), ("ols4", o), ("plain", prior))})
        seeds.append(view)
    return dict(recording=rec, rows=len(c), seeds=seeds)


def summarize(records, roster, previous):
    groups = sorted(set(roster.values()))
    boot = np.random.default_rng(target.SEED).integers(len(groups), size=(target.DRAWS, len(groups)))
    fields = ("cv", "ols4", "plain")
    by_group = {}
    for g in groups:
        z = [r for a in records if roster[a["recording"]] == g for r in a["seeds"]]
        by_group[g] = {k: {f: plain.merge_comparisons([s[k][f] for s in z])["completion_lower_pp"]
                           for f in fields} for k in ARMS}
    out = {}
    for k in ARMS:
        x = [v[k] for r in records for v in r["seeds"]]
        risk = [r["risk"] for r in x]
        signed = [r["known_easy_net_degradation"] for r in risk if r["known_easy_net_degradation"] is not None]
        m = dict(edits=sum(r["selected"] for r in risk), views=len(risk), risk_pass=sum(r["supported"] for r in risk),
            easy_over_2=sum(r["completion_easy_risk_upper"] is not None and r["completion_easy_risk_upper"] > .02 for r in risk),
            undefined_easy=sum(r["completion_easy_risk_upper"] is None for r in risk), worst_signed_easy=max(signed) if signed else None,
            dual_pass=sum(r["risk"]["supported"] and all(plain.merge_comparisons([r[f]])["completion_lower_pp"] > 0.
                         for f in ("cv", "ols4")) for r in x),
            ade=float(np.mean([r["ade"] for r in x if r["ade"] is not None])),
            fde=float(np.mean([r["fde"] for r in x if r["fde"] is not None])),
            worst_p95=max(r["p95"] for r in x if r["p95"] is not None))
        for f in fields:
            z = np.asarray([v[k][f] for v in by_group.values()])
            m[f"lower_vs_{f}_pp"] = dict(point=float(z.mean()), ci95=np.quantile(z[boot].mean(1), [.025, .975]).tolist(),
                                         positive_groups=int((z > 0.).sum()))
        out[k] = m
    ref, old = out["plain_gain"], previous["metrics"]["gain"]
    if ref["edits"] != old["edits"] or ref["risk_pass"] != old["supported"] or ref["dual_pass"] != old["both_floors_and_risk"]:
        raise ValueError("Frozen baseline interventions changed")
    for f in ("cv", "ols4"):
        np.testing.assert_allclose(ref[f"lower_vs_{f}_pp"]["point"], old[f"lower_gain_vs_{f}_pp"]["point"], atol=1e-12)
    return out, by_group


def score(key, manifest, old, roster, previous, replay):
    p = PUBLIC/"freeze.json"
    seal = run.read(p)
    if (seal["registration"] != key or seal["labels_used"] or seal["confirmation_access"]
            or [r["recording"] for r in seal["records"]] != [r["sequence"] for r in old["clips"]]):
        raise ValueError("Require the full freeze before scoring")
    run.checked(seal["training"])
    meta, provenance, cut = target.context(roster)
    records = []
    for i, (cell, f) in enumerate(zip(old["clips"], seal["records"], strict=True), 1):
        path = PRIVATE/f"{cell['sequence']}.score.json"
        if not path.exists():
            if replay:
                raise ValueError("Replay cannot read outcomes")
            row = score_recording(manifest, cell, f, meta, provenance, cut)
            run.once(path, run.stable(dict(registration=key, parent=f, row=row)))
        r = run.read(path)
        if r["registration"] != key or r["parent"] != f or r["row"]["rows"] != f["rows"]:
            raise ValueError("Wrong score receipt")
        records.append(r["row"])
        beat("replay" if replay else "score", completed=i, total=len(roster))
    m, groups = summarize(records, roster, previous)
    out = dict(registration=key, freeze=run.artifact(p), metrics=m, by_group=groups, recordings=len(records),
        rows=sum(r["rows"] for r in records), result_source="fresh_run", original_2pct=True, matched_count_only=True,
        calibration_labels_used_for_fit_or_thresholds=False, confirmation_access=False,
        independent_confirmation=False, deployment_changed=False)
    run.once(PUBLIC/"analysis.json", run.stable(out))
    print(m, flush=True)


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
        key, manifest, old, roster, parent, earlier, r, training = register()
        if phase == "train":
            train(key, training)
        elif phase == "freeze":
            freeze(key, manifest, old, roster, parent, earlier)
        else:
            score(key, manifest, old, roster, r, replay=phase == "replay")


if __name__ == "__main__":
    main()
