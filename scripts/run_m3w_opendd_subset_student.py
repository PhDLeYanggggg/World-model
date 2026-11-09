"""Learn when to keep a proposed edit from donor-only feasible subsets."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_opendd_optional_edits as optional
from src.evaluation import m3w_optional_subset_teacher as teacher

np, torch, study, base = optional.np, optional.torch, optional.study, optional.base
turn, slate, shared, risk = optional.turn, optional.slate, optional.shared, optional.risk
SITES, SEEDS = optional.SITES, optional.SEEDS
OUT = optional.OUT.parent/"opendd_subset_student_v1"
LOCAL = optional.LOCAL.parent/OUT.name
STEPS, EVERY, BATCH = 500, 50, 128
TARGETS = ("local_gain", "oracle_member", "forced_feasible")
ARMS = ("parent", "old_prefix", *TARGETS, "both", "matched_additive", "matched_joint")


def register():
    key, cohorts, parent, prior, _ = optional.register()
    files = [optional.OUT/f"{k}.json" for k in ("registration", "freeze", "readout")]
    f, r = map(study.read, files[1:])
    if (f["registration"] != key or r["freeze"] != study.artifact(files[1])
            or f["loaded_future_in_inference"] or r["reserved_outcomes_opened"]):
        raise ValueError("Require frozen optional-edit development controls")
    _, original, *_parents, refs, cal = turn.register()
    if original != cohorts:
        raise ValueError("Candidate producer source rows changed")
    p = OUT/"registration.json"
    base.immutable(p, dict(parents=[study.artifact(x) for x in files],
        code=[study.artifact(Path(x)) for x in (__file__, teacher.__file__, shared.costs.__file__,
                                               optional.__file__)],
        sites=SITES, seeds=SEEDS, steps=STEPS, batch=BATCH, checkpoint_every=EVERY,
        targets=TARGETS, roles=ARMS, original_2pct=True, observed_steps=8, forecast_steps=12,
        donors="past hindcast nomination and candidate medoid; neural producer excludes donor and outer site",
        features="same action geometry and current roster size; independent identical binary heads",
        thresholds="fixed zero logit; both requires both recording membership and forced feasibility",
        primary="keep/drop only on original held paths; paired reallocation diagnostic matches both radii",
        unresolved_labels="masked, not negative", development_only=True, independent_confirmation=False))
    return base.sha(p), cohorts, parent, prior, f, refs, cal


def beat(**fields):
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(dict(pid=os.getpid(), **fields))+"\n")
    p.replace(LOCAL/"heartbeat.json")


def proposals(d, bank):
    """A deterministic donor proposal; no cost head fit on this donor."""
    h, cv, ols, q = (d[k] for k in ("h", "cv", "ols", "q"))
    _, _, ok, hindcast = shared.costs.features(h, cv, ols, bank)
    p = cv.copy()
    # Choose the actual path nearest to the mean legal continuation.
    count = ok.sum(1)
    center = (bank*ok[..., None, None]).sum(1)/count.clip(1)[:, None, None]
    radius = np.linalg.norm(bank-center[:, None], axis=-1).mean(-1)
    arm = np.where(ok, radius, np.inf).argmin(1)
    for ix in slate.chunks(q):
        roster = np.arange(ix.start, ix.stop)
        good = roster[(hindcast[ix] > 0.) & (count[ix] > 0)]
        chosen = good[np.argsort(-hindcast[good], kind="stable")[:min(2, len(roster)-1)]]
        p[chosen] = bank[chosen, arm[chosen]]
    return p


def encode(d, path):
    x, _, live, _ = shared.costs.features(d["h"], d["cv"], d["ols"], path[:, None])
    size = np.log1p(np.bincount(d["q"])[d["q"]])
    x = np.column_stack((x[:, 0], size)).astype(np.float32)
    return x, live[:, 0]


def build_labels(d, path, cut):
    y = np.full((len(path), len(TARGETS)), np.nan, dtype=np.float32)
    counts = Counter()
    for rec in np.unique(d["rec"]):
        at = np.flatnonzero(d["rec"] == rec)
        q = np.unique(d["q"][at], return_inverse=True)[1]
        pack = teacher.build(d["cv"][at], d["ols"][at], path[at], d["y"][at], d["v"][at], q, cut)
        if pack is None:
            counts["empty"] += 1
            continue
        z, info = teacher.labels(pack)
        y[at[pack["chosen"]]] = z
        counts.update(info)
    return y, dict(counts)


def teacher_examples(key, held, donor, seed, d, ref, refs, cal, sources):
    pair = tuple(sorted((held, donor)))
    files = refs[f'{"+".join(pair)}_{seed}']
    cut = np.quantile(np.concatenate([sources[s]["e"] for s in SITES if s not in pair]), [.25, .75])
    ident = dict(registration=key, donor=donor, held=held, seed=seed, source=ref,
                 neural=files, cut=cut.tolist())
    path = LOCAL/f"{held}_{donor}_{seed}_labels.npz"
    if not path.exists():
        nn = turn.refit.pair_forecast(d["h"], pair, seed, refs, cal)
        bank = turn.append_turn(d["h"], np.concatenate((d["bank"], nn[:, None]), axis=1))
        p = proposals(d, bank)
        x, live = encode(d, p)
        y, info = build_labels(d, p, cut)
        turn.exp.previous.save_bank(path, dict(x=x[live], y=y[live], identity=study.json.dumps(ident, sort_keys=True),
                                             stats=study.json.dumps(info, sort_keys=True)))
    with np.load(path, allow_pickle=False) as z:
        if study.json.loads(z["identity"].item()) != ident:
            raise ValueError("Cached donor teacher lineage changed")
        x, y, counts = z["x"], z["y"], study.json.loads(z["stats"].item())
    if not len(x):
        return None, counts, study.artifact(path)
    counts["rows"] = len(x)
    counts.update({f"{k}_positive": int((y[:, i] == 1.).sum()) for i, k in enumerate(TARGETS)})
    return (torch.from_numpy(x), torch.from_numpy(y)), counts, study.artifact(path)


class Student(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = torch.nn.ModuleList(torch.nn.Sequential(
            torch.nn.Linear(shared.costs.WIDTH+1, 48), torch.nn.SiLU(),
            torch.nn.Linear(48, 24), torch.nn.SiLU(), torch.nn.Linear(24, 1)) for _ in TARGETS)
        for h in self.heads[1:]:
            h.load_state_dict(self.heads[0].state_dict())

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=1)


def fit(key, site, seed, data, refs):
    name = f"{site}_{seed}"
    identity = dict(registration=key, held=site, seed=seed, donors=refs)
    torch.manual_seed(seed)
    net = Student()
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    rng = np.random.default_rng(seed+4461)
    ckpt = LOCAL/f"{name}.pt"
    start, trace = 0, []
    if ckpt.exists():
        old = torch.load(ckpt, weights_only=True, map_location="cpu")
        if old["identity"] != identity or old["step"] not in range(EVERY, STEPS+1, EVERY):
            raise ValueError("Incompatible resume checkpoint")
        net.load_state_dict(old["net"])
        opt.load_state_dict(old["optimizer"])
        start, trace, rng.bit_generator.state = old["step"], old["trace"], old["rng"]
    ds = list(data.values())
    for step in range(start+1, STEPS+1):
        counts = rng.multinomial(BATCH, np.full(len(ds), 1./len(ds)))
        take = [rng.integers(len(v[0]), size=n) for v, n in zip(ds, counts, strict=True)]
        x, y = [torch.cat([v[k][a] for v, a in zip(ds, take, strict=True)]) for k in range(2)]
        valid = torch.isfinite(y)
        values = torch.nn.functional.binary_cross_entropy_with_logits(net(x), torch.nan_to_num(y), reduction="none")
        losses = (values*valid).sum(0)/valid.sum(0).clamp_min(1)
        loss = losses.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % EVERY == 0:
            trace.append(losses.detach().tolist())
            pending = ckpt.with_suffix(".pending")
            torch.save(dict(identity=identity, step=step, net=net.state_dict(), optimizer=opt.state_dict(),
                            trace=trace, rng=rng.bit_generator.state), pending)
            pending.replace(ckpt)
            beat(phase="fit", fold=name, step=step, total=STEPS)
    return net.eval(), dict(identity=identity, checkpoint=study.artifact(ckpt), loss=trace)


@torch.inference_mode()
def infer(net, relative, h, z, p):
    q, cv, ols, bank = (z[k] for k in ("q", "cv", "ols4", "bank"))
    parent = p["parent"]
    x, live = encode(dict(h=h, cv=cv, ols=ols, q=q), parent)
    logits = net(torch.from_numpy(x)).numpy()
    masks = {k: live & (logits[:, i] > 0.) for i, k in enumerate(TARGETS)}
    masks["both"] = masks["oracle_member"] & masks["forced_feasible"]
    paths = {k: np.where(m[:, None, None], parent, cv) for k, m in masks.items()}
    a, b = paths["oracle_member"].copy(), paths["oracle_member"].copy()
    changes = 0
    for ix in slate.chunks(q):
        radii = turn.turn.peak(a[ix]-cv[ix])
        if np.count_nonzero(radii) != 2:
            continue
        c, o, motion, hist = (x[ix] for x in (cv, ols, bank, h))
        paths0, row, arm, opts = slate.legal(c, motion, radii[radii > 0.])
        x, _, _, _ = shared.costs.features(hist, c, o, paths0)
        feat = np.column_stack((x[row, arm], np.full(len(row), np.log1p(len(c))))).astype(np.float32)
        unary = net(torch.from_numpy(feat)).numpy()[:, 1]
        value = unary[opts].sum(1)
        a[ix] = slate.assemble(c, paths0, row, arm, opts[int(value.argmax()):int(value.argmax())+1])[0]
        best = (-np.inf, None)
        # Rank pair gain in the same slate pool; normalize native gain by observed motion.
        unit = max(float(np.linalg.norm(np.diff(hist, axis=1), axis=-1).mean()), .05)
        for start in range(0, len(opts), 128):
            sel = opts[start:start+128]
            trial = slate.assemble(c, paths0, row, arm, sel)
            score = value[start:start+len(sel)]+slate.WEIGHT*slate.gain(relative, hist, c, o, trial, a[ix])[:, 2]/unit
            k = int(score.argmax())
            if score[k] > best[0]:
                best = (score[k], trial[k])
        b[ix] = best[1]
        changes += int(not np.allclose(a[ix], b[ix], atol=1e-9, rtol=1e-9))
    p_edit = turn.turn.peak(parent-cv)
    for k, v in paths.items():
        new = turn.turn.peak(v-cv)
        np.testing.assert_array_equal(new[new > 0.], p_edit[new > 0.])
    optional.validate(q, cv, p_edit, dict(parent=parent, local=paths["both"], prefix_additive=a, prefix_joint=b))
    for ix in slate.chunks(q):
        get = lambda p: np.sort(turn.turn.peak(p[ix]-cv[ix]))
        np.testing.assert_allclose(get(a), get(paths["oracle_member"]), atol=1e-9, rtol=1e-9)
    return dict(q=q, cv=cv, ols4=ols, parent=parent, old_prefix=p["prefix_joint"], **paths,
                matched_additive=a, matched_joint=b, logits=logits), changes


def freeze(key, cohorts, parent, prior, old, refs, cal):
    data = turn.exp.loader(cohorts)
    checkpoints, bank, counts = {}, {}, {}
    for site in SITES:
        h, q, cv, ols, _ = turn.menu.parent.source_inputs(cohorts[site])
        for seed in SEEDS:
            name = f"{site}_{seed}"
            donors, receipt, coverage = {}, {}, {}
            for s in SITES:
                if s == site:
                    continue
                ex, coverage[s], artifact = teacher_examples(key, site, s, seed, data[s], cohorts[s], refs, cal, data)
                if ex is not None:
                    donors[s], receipt[s] = ex, artifact
                beat(phase="teacher", fold=name, donor=s, completed=len(checkpoints))
            if not donors:
                raise ValueError("No donor candidate examples")
            net, checkpoints[name] = fit(key, site, seed, donors, receipt)
            relative = slate.model.RelativeGain().eval()
            relative.load_state_dict(risk.state(prior["fits"][name], site, slate.STEPS))
            with (np.load(study.checked(parent["banks"][name]), allow_pickle=False) as z,
                  np.load(study.checked(old["forecasts"][name]), allow_pickle=False) as p):
                for k, arr in (("q", q), ("cv", cv), ("ols4", ols)):
                    np.testing.assert_array_equal(z[k], arr)
                    np.testing.assert_array_equal(p[k], arr)
                forecasts, changes = infer(net, relative, h, z, p)
            bank[name] = turn.exp.previous.save_bank(LOCAL/f"{name}.npz", forecasts)
            counts[name] = dict(donors=coverage, matched_joint_changes=changes)
            beat(phase="freeze", fold=name, completed=len(checkpoints), total=21)
            print(name, changes, flush=True)
    base.immutable(OUT/"freeze.json", dict(registration=key, fits=checkpoints, forecasts=bank,
        counts=counts, inference_past_only=True, held_scoring_not_run=True))


def records(d, p, cut):
    rows = []
    for rec in np.unique(d["rec"]):
        at = d["rec"] == rec
        q = np.unique(d["q"][at], return_inverse=True)[1]
        cv, ols, y, v = (d[k][at] for k in ("cv", "ols", "y", "v"))
        w = 1./np.bincount(q)[q]
        mass = float(w @ np.where(v, np.linalg.norm(cv-y, axis=-1), 0.).sum(1))
        path = {k: p[k][at] for k in ARMS}
        arms = {}
        for k, z in path.items():
            s = study.policy(z, cv, cv, y, v, ["r"]*len(cv), easy_cut=cut[0], hard_cut=cut[1])
            g = [float(100.*(w@turn.exact.headroom.lower(a, z, y, v))/mass) for a in (cv, ols)] if mass else None
            arms[k] = dict(risk=s["recordings"]["r"], gain_pp=g, ade=s["known_ADE_pixel"],
                fde=s["available_endpoint_FDE_diagnostic_pixel"], tail=s["p95_known_ADE_pixel"],
                hard=s["subsets"]["hard"]["gain_percent"])
        effect = turn.expanded.actual_contrast(cv, ols, path["matched_joint"], path["matched_additive"], y, v, q, cut)
        rows.append(dict(record=str(rec), arms=arms, matched=effect))
    return rows


def metrics(rows):
    result = {}
    for k in ARMS:
        samples = [r["arms"][k] for r in rows]
        edited = [v for v in samples if v["risk"]["selected"]]
        result[k] = dict(views=len(samples), edited_views=len(edited), edits=sum(v["risk"]["selected"] for v in edited),
            risk_pass=sum(v["risk"]["supported"] for v in edited),
            undefined_easy_edited=sum(v["risk"]["completion_easy_risk_upper"] is None for v in edited),
            signed_easy_over_2pct=sum((v["risk"]["known_easy_net_degradation"] or 0.) > .02 for v in samples),
            dual=sum(v["risk"]["supported"] and v["gain_pp"] is not None and min(v["gain_pp"]) > 1e-8 for v in edited),
            by_record_gain=turn.effects.site_means(
                [(r["record"], r["arms"][k]["gain_pp"]) for r in rows if r["arms"][k]["gain_pp"] is not None]))
        for m in ("ade", "fde", "tail", "hard"):
            vals = [v[m] for v in samples if v[m] is not None]
            result[k][m] = float(np.mean(vals)) if vals else None
    result["matched"] = turn.effects.summary(
        [dict(record=r["record"], effect=r["matched"]["pair_native"], qualified=r["matched"]["qualified"])
         for r in rows if r["matched"] is not None])
    return result


def probe(y, logits):
    from sklearn.metrics import average_precision_score, roc_auc_score
    out = {}
    for i, k in enumerate(TARGETS):
        live = np.isfinite(y[:, i])
        a, b = y[live, i], logits[live, i]
        n = int(a.sum())
        pred = b > 0.
        out[k] = dict(rows=len(a), positives=n, threshold_selected=int(pred.sum()),
            precision=float(a[pred].mean()) if pred.any() else None,
            recall=float(a[pred].sum()/n) if n else None,
            average_precision=float(average_precision_score(a, b)) if n else None,
            auroc=float(roc_auc_score(a, b)) if 0 < n < len(a) else None)
    return out


def score(key, cohorts, old):
    path = OUT/"freeze.json"
    f = study.read(path)
    if f["registration"] != key or not f["held_scoring_not_run"] or len(f["forecasts"]) != 21:
        raise ValueError("Freeze every held fold before scoring")
    data = turn.exp.loader(cohorts)
    by_site, all_rows, receipts, labels, logits = {}, [], {}, [], []
    for s in SITES:
        d = data[s]
        cut = np.quantile(np.concatenate([data[t]["e"] for t in SITES if t != s]), [.25, .75])
        rows, ay, ax = [], [], []
        for seed in SEEDS:
            name = f"{s}_{seed}"
            fit = f["fits"][name]["identity"]
            if fit["held"] != s or s in fit["donors"]:
                raise ValueError("Scored site entered student training")
            bind = dict(registration=key, predictions=f["forecasts"][name], baseline=old["forecasts"][name])
            cache = LOCAL/f"{name}_score.json"
            if not cache.exists():
                with (np.load(study.checked(bind["predictions"]), allow_pickle=False) as p,
                      np.load(study.checked(bind["baseline"]), allow_pickle=False) as z):
                    for k, v in (("q", d["q"]), ("cv", d["cv"]), ("ols4", d["ols"]),
                                 ("parent", z["parent"]), ("old_prefix", z["prefix_joint"])):
                        np.testing.assert_array_equal(p[k], v)
                    y, counts = build_labels(d, p["parent"], cut)
                    live = (p["parent"] != p["cv"]).any((1, 2))
                    estimates = {k: p[k] for k in ARMS}
                    pr = turn.exp.previous.save_bank(LOCAL/f"{name}_probe.npz", dict(y=y[live], logits=p["logits"][live]))
                    base.immutable(cache, dict(binding=bind, rows=records(d, estimates, cut), probe=pr, solver=counts))
            saved = study.read(cache)
            if saved["binding"] != bind:
                raise ValueError("Scoring cache identity changed")
            with np.load(study.checked(saved["probe"]), allow_pickle=False) as pr:
                ay.append(pr["y"])
                ax.append(pr["logits"])
            rows.extend(saved["rows"])
            receipts[name] = study.artifact(cache)
            beat(phase="score", fold=name, completed=len(receipts), total=21)
            print(name, len(receipts), flush=True)
        labels.extend(ay)
        logits.extend(ax)
        by_site[s] = dict(**metrics(rows), probe=probe(np.concatenate(ay), np.concatenate(ax)))
        all_rows.extend({**r, "record": f"{s}:{r['record']}"} for r in rows)
    total = metrics(all_rows)
    for k in ARMS:
        total[k].pop("by_record_gain")
        total[k]["location_bootstrap_gain_pp"] = turn.effects.interval([v[k]["by_record_gain"] for v in by_site.values()])
    total["matched"].pop("site_mean_native")
    total["matched"]["location_bootstrap_native"] = turn.effects.interval([v["matched"]["site_mean_native"] for v in by_site.values()])
    if (total["parent"]["edits"], total["parent"]["risk_pass"], total["old_prefix"]["edits"]) != (3690, 47, 282):
        raise ValueError("The optional-edit controls did not replay")
    if len({total[k]["edits"] for k in ("oracle_member", "matched_additive", "matched_joint")}) != 1:
        raise ValueError("Matched utility comparison changed edit count")
    base.immutable(OUT/"readout.json", dict(registration=key, freeze=study.artifact(path),
        totals=total, prediction_probe=probe(np.concatenate(labels), np.concatenate(logits)),
        sites=by_site, checkpoints=receipts, new_fold_fits=21, independent_heads=63,
        result_source="fresh_run", original_2pct=True, reserved_outcomes_opened=False, independent_confirmation=False))
    print(study.json.dumps(total, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score"))
    phase = parser.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for d in (OUT, LOCAL):
        d.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, cohorts, parent, prior, old, refs, cal = register()
        if phase == "freeze":
            freeze(key, cohorts, parent, prior, old, refs, cal)
        else:
            score(key, cohorts, old)


if __name__ == "__main__":
    main()
