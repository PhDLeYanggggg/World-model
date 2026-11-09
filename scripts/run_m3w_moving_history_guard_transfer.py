"""Transfer frozen DroneCrowd history calibration to exposed Moving development."""

import argparse
import fcntl
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_recording_utility_guard as utility
from scripts import run_m3w_moving_dual_record_forecast as moving
from src.data_unification import m3w_moving_forecast_data as prep


stage, np, run = utility.stage, utility.np, utility.run
prior, torch = stage.parent.prior, stage.parent.torch
PUBLIC = utility.PUBLIC.parent/"moving_history_guard_transfer_v1"
PRIVATE = utility.PRIVATE.parent/PUBLIC.name
ADMISSION = moving.OUT.parent/"moving_development_admission_v1"/"admission.json"
STRUCTURE = moving.OUT.parent/"movingdronecrowd_intake_v1"/"analysis.json"
GATES = ("raw_gain_risk_point", "history_gain_lower_risk_point", "history_gain_lower_log_risk")


def register():
    _, _, _, _, roster, source = utility.register()
    p = utility.PUBLIC/"analysis.json"
    a = run.read(p)
    dev = moving.OUT/"freeze.json"
    b = run.read(dev)
    if a["confirmation_access"] or not b["before_held_scoring"]:
        raise ValueError("Require only frozen development data")
    cuts = {}
    for cell in b["cells"]:
        for g in cell["identity"]["held"]:
            cut = tuple(cell["cut"])
            if g in cuts and cuts[g] != cut:
                raise ValueError("Inconsistent legacy easy cut")
            cuts[g] = cut
    train = prior.PUBLIC/"training.json"
    fits = run.read(train)["fits"]
    donors = set(fits[0]["identity"]["donors"])
    if len(cuts) != 9 or len(donors) != 14 or donors & set(cuts) or any(set(f["identity"]["donors"]) != donors for f in fits):
        raise ValueError("Target source must be disjoint from all neural training sources")
    admitted = prep.load_admission(ROOT, ADMISSION)
    meta = {r["clip"]: r for r in run.read(STRUCTURE)["recordings"]}
    records = []
    for node, r in sorted(admitted["recordings"].items()):
        group = r["exclusion_group"]
        if group in cuts:
            m = meta[r["clip"]]
            n = m["window_support"]["obs8_pred12_native_stride1"]["history_complete_agent_instances"]
            if n:
                records.append(dict(node=node, clip=r["clip"], source=group, cut=cuts[group], rows=n, frames=m["image_frames"]))
    if len(records) != 24 or {r["source"] for r in records} != set(cuts):
        raise ValueError("Require the original 24 Moving development recordings")
    reg = PUBLIC/"registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), source_calibration=run.artifact(p),
        source_fit=run.artifact(train), target_definition=[run.artifact(k) for k in (dev, ADMISSION, STRUCTURE)],
        observations=8, forecast=12, seed=prior.SEEDS, records=records, donors=sorted(donors), guards=GATES,
        target_role="already exposed Moving B development sources; no Moving A or new target outcome role",
        procedure="freeze analytic bank and independent choice with prior weights; history-only calibration fits all 15 DroneCrowd sources; "
                  "use maximum source-held residuals for the log risk and utility bounds; no Moving labels fit any coefficient",
        support="undefined observed-prefix risk or gain implies abstention; all recordings remain in the score",
        scope="full-scene OLS4-default diagnostic; not a matched sparse peak-budget comparison",
        original_2pct=True, independent_confirmation=False, confirmation_access=False)))
    return run.sha(reg), roster, source, admitted, records


def read_record(admission, r):
    meta = admission["recordings"][r["node"]]
    path = ROOT/meta["source_annotation_path"]
    prep.require_fit_record(admission, r["node"], "fit", path)
    return prep.mot_record(path.read_bytes(), r["clip"], range(1, r["frames"]+1))


def inputs(raw):
    for frame in range(7, raw.valid.shape[1]-12):
        p = prep.past_inputs(raw, frame, prep.SPEC)
        at = p["history_complete"]
        if at.any():
            yield frame, p, at, p["positions"][at]


@torch.inference_mode()
def forecast(raw, nets, easy_cut):
    blocks, ids = [], []
    risk_mass, donor_mass = np.zeros((len(nets), 4)), np.zeros((len(nets), 4))
    history_gain, raw_gain = np.zeros(len(nets)), np.zeros((len(nets), 2))
    hist_ref = 0.
    for f, p, at, h in inputs(raw):
        a, o, paths, scale = prior.model.features(h)
        legal = (paths[:, 1:] != paths[:, 1:2]).any((2, 3))
        changed = (paths[:, 1:] != paths[:, :1]).any((2, 3))
        aa, oo = torch.from_numpy(a), torch.from_numpy(o)
        choices = []
        for i, net in enumerate(nets):
            v, refs = net(aa, oo)
            v, refs = v.numpy()*scale[:, None, None], refs.numpy()*scale[:, None]
            gain = v[..., 0]-v[:, :1, 0]
            x = stage.parent.parent.study.choose(gain, legal)
            choices.append(x)
            raw_gain[i] += (gain[np.arange(len(x)), x].mean(), refs[:, 0].mean())
            row = np.arange(len(x))
            take = changed[row, x]
            donor_mass[i] += np.r_[v[row, x, 1:][take].sum(0), refs[take].sum(0)]
        x = np.column_stack(choices).astype(np.uint8)
        risk_mass += stage.study.prefix_masses(h, x[..., None], changed, easy_cut)[:, 0]
        q, ref = utility.study.prefix_utility(h, x)
        history_gain += q
        hist_ref += ref
        blocks.append(x)
        ids.append(np.column_stack((np.full(len(x), f), p["agent_ids"][at])))
    ratio = lambda a: [[float(x/y) if y > 0. else None for x, y in zip(v[:2], v[2:], strict=True)] for v in a]
    return (np.concatenate(blocks), prep.array_hash(np.concatenate(ids)), ratio(risk_mass), ratio(donor_mass),
            [float(g/hist_ref*100.) if hist_ref > 0. else None for g in history_gain],
            [float(g/r*100.) if r > 0. else None for g, r in raw_gain])


def fit(known_x, known_y, groups, new_x, estimator, log=False):
    """No target outcomes enter the estimator; target-shaped placeholders only."""
    outer = np.arange(len(groups))
    new = np.arange(len(groups), len(groups)+len(new_x))
    y = np.concatenate((known_y, np.zeros((len(new), *known_y.shape[1:]))))
    x = np.concatenate((known_x, new_x))
    g = np.r_[groups, np.repeat("moving_transfer_target", len(new))]
    point = estimator(x, y, g, outer, new)
    margin = []
    for name in np.unique(groups):
        tr, te = outer[groups != name], outer[groups == name]
        p = estimator(x, y, g, tr, te)
        error = np.log1p(100.*y[te])-np.log1p(100.*p) if log else p-y[te]
        margin.append(error.max(0))
    gap = np.maximum(np.max(margin, axis=0), 0.)
    bound = np.expm1(np.log1p(100.*point)+gap)/100. if log else point-gap
    return point, bound, np.asarray(gap).tolist()


def source_labels(roster, source):
    src_key, *_ = stage.register()
    (_, h, risk, _, groups, _, _), _ = stage.read_frozen(src_key, roster, source)
    f = run.read(run.checked(run.read(utility.PUBLIC/"analysis.json")["freeze"]))
    gains, obs = [], []
    for a, b in zip(f["recordings"], source["recordings"], strict=True):
        r = run.read(stage.parent.PRIVATE/f"{b['recording']}.score.json")
        if a["recording"] != b["recording"] or a["choices"] != b["choices"] or r["freeze"] != b:
            raise ValueError("Invalid source calibration alignment")
        gains.extend(a["gain_pp"])
        obs.extend(row[0]["actual"]["ols4"] for row in r["observed"])
    return h[:, :1, :, None], risk[:, :1], np.asarray(gains)[:, None], np.asarray(obs), groups


def freeze(key, roster, source, admission, records):
    nets = prior.load_nets(run.sha(prior.PUBLIC/"registration.json"))
    rows = []
    for i, r in enumerate(records, 1):
        p = PRIVATE/f"{i:02d}.json"
        if not p.exists():
            x, ids, h, donor, g, raw = forecast(read_record(admission, r), nets, r["cut"][0])
            if len(x) != r["rows"]:
                raise ValueError("Past-visible roster changed")
            c = stage.target.save_array(PRIVATE/f"{i:02d}.npy", x)
            run.once(p, run.stable(dict(registration=key, recording=r["node"], choices=c, row_hash=ids,
                                          history_risk=h, donor_risk=donor, history_gain=g, raw_gain=raw)))
        z = run.read(p)
        if z["registration"] != key or z["recording"] != r["node"]:
            raise ValueError("Invalid Moving receipt")
        run.checked(z["choices"])
        rows.append(z)
        if i % 6 == 0:
            print(f"{i}/{len(records)} Moving observed-only recordings", flush=True)
    h, y, u, v, groups = source_labels(roster, source)
    x = np.asarray([z["history_risk"] for z in rows], dtype=float).reshape(-1, 1, 2, 1)
    a = np.asarray([z["history_gain"] for z in rows], dtype=float).reshape(-1, 1)
    risk_ok, gain_ok = np.isfinite(x).all((1, 2, 3)), np.isfinite(a[:, 0])
    p, upper, gp, lower = np.full((len(a), 2), np.nan), np.full((len(a), 2), np.nan), np.full(len(a), np.nan), np.full(len(a), np.nan)
    pp, uu, margin = fit(h, y, groups, x[risk_ok], stage.study.fit_predict, log=True)
    p[risk_ok], upper[risk_ok] = pp[:, 0], uu[:, 0]
    gp[gain_ok], lower[gain_ok], gap = fit(u, v, groups, a[gain_ok], utility.study.fit_predict)
    raw = np.asarray([z["raw_gain"] for z in rows], dtype=float).ravel()
    gates = np.column_stack(((p <= .02).all(1) & (raw > 0.), (p <= .02).all(1) & (lower > 0.),
                             (upper <= .02).all(1) & (lower > 0.)))
    serial = lambda z: np.where(np.isfinite(z), z, None).tolist()
    for i, z in enumerate(rows):
        s = slice(i*len(nets), (i+1)*len(nets))
        z.update(risk_point=serial(p[s]), risk_upper=serial(upper[s]), gain_point=serial(gp[s]),
                 gain_lower=serial(lower[s]), gates=gates[s].tolist())
    run.once(PUBLIC/"freeze.json", run.stable(dict(registration=key, records=rows, risk_log_margin=margin,
        utility_margin_pp=gap, calibration_sources=len(np.unique(groups)), predicted_risk_supported=int(risk_ok.sum()),
        predicted_gain_supported=int(gain_ok.sum()), moving_labels_used=False, confirmation_access=False)))


def score_record(raw, r, f):
    banks, labels, masks, q, ids = [], [], [], [], []
    for frame, p, at, h in inputs(raw):
        banks.append(prior.model.floor.paths(prior.model.floor.base(h)))
        y = prep.target_labels(raw, p, prep.SPEC)
        v = y["future_mask"][at]
        labels.append(np.where(v[..., None], y["future_positions"][at]-h[:, -1:], 0.))
        masks.append(v)
        q.extend([frame]*len(h))
        ids.append(np.column_stack((np.full(len(h), frame), p["agent_ids"][at])))
    x = np.load(run.checked(f["choices"]), mmap_mode="r", allow_pickle=False)
    if prep.array_hash(np.concatenate(ids)) != f["row_hash"] or len(x) != r["rows"]:
        raise ValueError("Scoring roster differs from freeze")
    bank, y, v = map(np.concatenate, (banks, labels, masks))
    cv, ols = bank[:, 0], bank[:, 1]
    _, inv, count = np.unique(q, return_inverse=True, return_counts=True)
    w, group = 1./count[inv], np.zeros(len(x), dtype=int)
    def one(path):
        m = prior.policy(path, cv, cv, y, v, group, easy_cut=r["cut"][0], hard_cut=r["cut"][1])
        risk = m["recordings"]["0"]
        pairs = [prior.comparison_masses(ref, path, cv, y, v, w) for ref in (cv, ols)]
        gains = [prior.merge_comparisons([d])["completion_lower_pp"] for d in pairs]
        return dict(risk=risk["supported"], easy_defined=risk["completion_easy_risk_upper"] is not None,
                    risk_ratios=[risk["completion_all_risk_upper"], risk["completion_easy_risk_upper"]],
                    both=risk["supported"] and all(g is not None and g > 0. for g in gains), gains=gains, masses=pairs,
                    ade=m["known_ADE_pixel"], fde=m["available_endpoint_FDE_diagnostic_pixel"], p95=m["p95_known_ADE_pixel"],
                    cv_edits=risk["selected"])
    return dict(cv=one(cv), ols4=one(ols), fitted=one(bank[:, 3]),
                independent=[one(bank[np.arange(len(x)), x[:, j]+1]) for j in range(x.shape[1])],
                rows=len(x), known=int(v.any(1).sum()), complete=int(v.all(1).sum()))


def summarize(rows, freeze, records):
    names = sorted({r["source"] for r in records})
    draws = np.random.default_rng(stage.target.SEED).integers(len(names), size=(stage.target.DRAWS, len(names)))
    menu, guard = {}, {}
    for a in ("cv", "ols4", "fitted", "independent"):
        by_source = {s: [m for r, z in zip(records, rows, strict=True) if r["source"] == s
                         for m in (z[a] if a == "independent" else [z[a]])] for s in names}
        full = [m for z in by_source.values() for m in z]
        effect = np.array([prior.merge_comparisons([m["masses"][1] for m in z])["completion_lower_pp"] for z in by_source.values()])
        menu[a] = dict(views=len(full), risk_pass=sum(m["risk"] for m in full), both=sum(m["both"] for m in full),
            ade=float(np.mean([m["ade"] for m in full if m["ade"] is not None])),
            fde=float(np.mean([m["fde"] for m in full if m["fde"] is not None])),
            worst_p95=max(m["p95"] for m in full if m["p95"] is not None),
            ols4_lower_pp=float(effect.mean()), ols4_ci95=np.quantile(effect[draws].mean(1), [.025, .975]).tolist())
    for j, g in enumerate(GATES):
        on = [(r["source"], m) for r, z, f in zip(records, rows, freeze["records"], strict=True)
              for m, gate in zip(z["independent"], f["gates"], strict=True) if gate[j]]
        guard[g] = dict(selected=len(on), sources=len({s for s, _ in on}), risk_pass=sum(m["risk"] for _, m in on),
                        both=sum(m["both"] for _, m in on), abstentions=len(records)*len(prior.SEEDS)-len(on))
    groups = np.repeat([r["source"] for r in records], len(prior.SEEDS))
    pred = lambda k: np.asarray([v for f in freeze["records"] for v in f[k]], dtype=float)
    actual = [v for z in rows for v in z["independent"]]
    risk = 100.*np.asarray([v["risk_ratios"] for v in actual], dtype=float)
    gain = np.asarray([v["gains"][1] for v in actual], dtype=float)
    calibration = {}
    for name, old, new, y in (("all_risk", 100.*pred("donor_risk")[:, 0], 100.*pred("risk_point")[:, 0], risk[:, 0]),
                               ("easy_risk", 100.*pred("donor_risk")[:, 1], 100.*pred("risk_point")[:, 1], risk[:, 1]),
                               ("utility", pred("raw_gain"), pred("gain_point"), gain)):
        valid = np.isfinite(old) & np.isfinite(new) & np.isfinite(y)
        use = sorted(set(groups[valid]))
        mae = np.array([[np.abs(v-y)[valid & (groups == g)].mean() for g in use] for v in (old, new)])
        calibration[name] = dict(common_views=int(valid.sum()), common_sources=len(use),
            raw_donor_mae_pp=float(mae[0].mean()) if use else None, history_mae_pp=float(mae[1].mean()) if use else None)
    return dict(menu=menu, guards=guard, sources=len(names), recordings=len(rows), rows=sum(r["rows"] for r in rows),
                known=sum(r["known"] for r in rows), complete=sum(r["complete"] for r in rows),
                calibration=calibration, predicted_risk_supported=freeze["predicted_risk_supported"],
                predicted_gain_supported=freeze["predicted_gain_supported"])


def score(key, admission, records, replay):
    p = PUBLIC/"freeze.json"
    f = run.read(p)
    if f["registration"] != key or f["moving_labels_used"] or f["confirmation_access"]:
        raise ValueError("Require outcome-free Moving decisions")
    rows = []
    for i, (r, z) in enumerate(zip(records, f["records"], strict=True), 1):
        mark = PRIVATE/f"{i:02d}.score.json"
        if not mark.exists():
            if replay:
                raise ValueError("Replay may only use score summaries")
            if r["node"] != z["recording"]:
                raise ValueError("Wrong frozen recording")
            t = score_record(read_record(admission, r), r, z)
            run.once(mark, run.stable(dict(freeze=z, scores=t)))
        s = run.read(mark)
        if s["freeze"] != z:
            raise ValueError("Incorrect outcome receipt")
        rows.append(s["scores"])
        if i % 6 == 0:
            print(f"{i}/{len(records)} Moving score summaries", flush=True)
    report = summarize(rows, f, records)
    run.once(PUBLIC/"analysis.json", run.stable(dict(**report, freeze=run.artifact(p), result_source="fresh_run",
        calibration_fit_moving_outcomes=False, original_2pct=True, independent_confirmation=False,
        confirmation_access=False, deployment_changed=False)))
    print(report, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "score", "replay"))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for d in (PUBLIC, PRIVATE):
        d.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, roster, source, admission, records = register()
        if phase == "freeze":
            freeze(key, roster, source, admission, records)
        else:
            score(key, admission, records, replay=phase == "replay")


if __name__ == "__main__":
    main()
