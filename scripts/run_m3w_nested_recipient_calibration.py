"""Fit, calibrate and evaluate on disjoint source groups inside development."""

import argparse
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

if sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "4"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_recipient_feasibility as prior
from src.evaluation.m3w_grouped_recording_readout import grouped_readout
from src.evaluation.m3w_recording_switch_ceiling import moments
from src.evaluation.m3w_selection_forecast_readout import errors
from src.world_model import m3w_nested_recipient_calibration as model
from sklearn.metrics import average_precision_score


run, np, torch = prior.run, prior.np, prior.torch
OUT = prior.OUT.parent/"nested_recipient_calibration_v1"
PRIVATE = prior.PRIVATE.parent/OUT.name
CV, OLS, SEEDS = prior.CV, prior.OLS, prior.SEEDS
STEPS, EVERY, BATCH = 500, 100, 128
ARMS = ("fixed_support", "fixed_gain", "cal_support", "cal_gain", "matched_cal_gain")
PAIRS = {f"{a}_vs_{'cv' if b == CV else b}": (b, a, CV) for a, b in (
    (OLS, CV), *((a, b) for a in ARMS for b in (CV, OLS)),
    ("fixed_support", "fixed_gain"), ("cal_support", "cal_gain"), ("cal_support", "matched_cal_gain"))}


def register():
    parent, identity, cut, _ = prior.register()
    file = OUT/"registration.json"
    registered = run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parent=parent, parent_diagnostic=run.artifact(prior.OUT/"search_audit.json"),
        source_roles="six fitting, next+fourth cyclic calibration, one held; all original fitting groups",
        threshold_grid=model.THRESHOLDS, original_2pct=True, seeds=SEEDS,
        steps=STEPS, batch=BATCH, checkpoint_every=EVERY, lr=.001, cpu_threads=4, workers=0,
        nominee="six-group ridge costs, fixed gain minus four times both harms",
        local_quotas_only=True, early_stop_or_outer_selection=False, official_calibration_access=False,
        confirmation_access=False, deployment=False))
    if file.exists() and (before := run.read(file)) != registered:
        old_sha = "cfb6dfbb971cd0e4405ea40b2fbfa26ee70ee3449df766d4fa8cd38b3a902668"
        if before["code"]["sha256"] != old_sha or dict(before, code=registered["code"]) != registered:
            raise ValueError("Only the reporting import may differ from the original run")
        run.once(OUT/"reporting_patch.json", run.stable(dict(original=run.artifact(file),
            patched=registered["code"], scope="final_grouped_readout_import_only_after_all_score_cells_written",
            changes_to_fit_calibration_or_predictions=False)))
    else:
        run.once(file, registered)
    return run.sha(file), identity, cut


def nominee(fit, cut, x, targets, blocks):
    b, gain, known, raw = targets
    x, radius = (v.numpy() for v in x)
    easy = known & (raw > 0.) & (raw <= cut)
    accum = {g: [] for g in fit}
    for z in blocks:
        if z["source"] not in fit:
            continue
        s, w = z["sl"], z["weight"]
        scaled = np.column_stack((b[s], easy[s]*b[s], ~known[s]))
        effect = gain[s]
        harm = np.maximum(-effect, 0.)
        output = np.concatenate((np.stack((effect, harm, easy[s, None]*harm), axis=2),
            np.broadcast_to(scaled[:, None], (*effect.shape, 3))), axis=2)
        q = x[s].swapaxes(0, 1).astype(float)
        weighted = q*w[None, :, None]
        accum[z["source"]].append((q.transpose(0, 2, 1)@weighted,
            weighted.transpose(0, 2, 1)@output.swapaxes(0, 1), w.sum()))
    fit_moments = []
    for g in fit:
        gram, rhs, w = (sum(v) for v in zip(*accum[g], strict=True))
        fit_moments.append((gram/w, rhs/w))
    coef = prior.prior.old.model.coefficients(fit_moments)
    score = []
    for z in blocks:
        s = z["sl"]
        values = prior.prior.old.model.decision_costs(x[s], coef, z["bank"], z["cv"], z["scale"])
        benefit, all_excess, easy_excess, legal = values
        # Same geometry and cost penalty for both recipient rankers.
        nom = np.where(legal, benefit-4.*(all_excess+easy_excess), -np.inf).argmax(1)
        ix = np.arange(len(nom)), nom
        score.append((nom, legal.any(1), benefit[ix]/z["scale"]))
    return coef, score


def fit_model(binding, held, seed, fit, cal, easy_cut, x, y, groups, legal, beat, replay):
    path = PRIVATE/f"fit_{held}_{seed}.pt"
    key = dict(registration=binding, held=held, seed=seed, fit=fit, calibration=cal, easy_cut=easy_cut)
    torch.manual_seed(seed)
    net = model.SupportRanker(x.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    rng = torch.Generator().manual_seed(seed+7919)
    ids = [groups[g][legal[groups[g]]] for g in fit]
    step, trace = 0, []
    if path.exists():
        s = torch.load(path, map_location="cpu", weights_only=True)
        if s["key"] != key or not 0 < s["step"] <= STEPS:
            raise ValueError("Checkpoint disagrees with registered split")
        net.load_state_dict(s["model"])
        opt.load_state_dict(s["optimizer"])
        rng.set_state(s["rng"])
        step, trace = s["step"], s["trace"]
    if replay and step != STEPS:
        raise ValueError("Replay may not train")
    for i in range(step, STEPS):
        ix = prior.prior.prior.base.fit.draw_batch(ids, BATCH, rng)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x[ix]), y[ix])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite support training")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if (i+1) % EVERY == 0:
            trace.append(float(loss.detach()))
            temp = path.with_suffix(".pending")
            torch.save(dict(key=key, model=net.state_dict(), optimizer=opt.state_dict(), rng=rng.get_state(),
                step=i+1, trace=trace), temp)
            temp.replace(path)
            beat(stage="checkpoint", held=held, seed=seed, step=i+1, loss=trace[-1])
    return net.eval(), run.artifact(path)


def calibration_cell(b, p, cut):
    """Fixed nominee moments; only used for calibration groups."""
    m = moments(b["cv"], p, b["target"], b["valid"], cut["easy"])
    ols = errors(b["ols"], b["target"], b["valid"])[0]
    w, known = b["weight"], m["known"]
    base_ols = np.linalg.norm(b["ols"]-b["cv"], axis=2).max(1)
    alternative = np.linalg.norm(b["ols"]-p, axis=2).max(1)
    offset = w@np.where(known, ols-m["base"], -base_ols)
    change = np.where(known, m["diff"], base_ols-alternative)
    denominator = float(w@m["base"])
    return m, float(offset), change, denominator


def trial(blocks, scores, choices, t):
    count, ok, by_source = 0, 0, {}
    for b, score, (legal, cell) in zip(blocks, scores, choices, strict=True):
        on = model.pick(score, legal, b["inverse"], b["counts"], t)
        m, offset, change, denominator = cell
        if denominator <= 0.:
            raise ValueError("Calibration must retain observed reference mass")
        w = b["weight"][on]
        count += int(on.sum())
        ok += bool(m["ref_easy"][on].sum() > 0. and m["all_excess"][on].sum() <= 0.
                   and m["easy_excess"][on].sum() <= 0.
                   and m["easy_signed"][on].sum() <= .02*m["ref_easy"].sum())
        by_source.setdefault(b["source"], []).append(np.array([w@m["utility"][on], offset+w@change[on],
                                                                denominator]))
    cells = [np.sum(s, axis=0) for s in by_source.values()]
    gains = np.mean([100*c[:2]/c[2] for c in cells], axis=0)
    return dict(threshold=t, views=len(blocks), passes=ok, switches=count, gain_both=gains.tolist())


@torch.inference_mode()
def predict(net, x):
    return torch.cat([net.score(a) for a in x.split(4096)]).numpy()


def freeze(binding, data, cut, beat, replay):
    x, truth, blocks = data
    groups = prior.prior.group_ids(blocks)
    b, gain, known, raw = truth
    marks, calibration = [], []
    for held in groups:
        fit, cal = model.roles(groups, held)
        ids = np.concatenate([groups[g] for g in fit])
        train_raw = raw[ids]
        local_cut = float(np.quantile(train_raw[np.isfinite(train_raw) & (train_raw > 0.)], .25))
        nom_path = PRIVATE/f"nomination_{held}.json"
        if not nom_path.exists():
            if replay:
                raise ValueError("Replay may not fit the path nominee")
            coef, choices = nominee(fit, local_cut, x, truth, blocks)
            run.once(nom_path, run.stable(dict(registration=binding, held=held, fit=fit, cal=cal,
                coefficient=coef.tolist(), nominations=[(a.tolist(), ok.tolist(), g.tolist()) for a, ok, g in choices])))
        doc = run.read(nom_path)
        if (doc["registration"], doc["held"], doc["fit"], doc["cal"]) != (binding, held, fit, cal):
            raise ValueError("Path nominee uses the wrong source roles")
        nom, legal, gain_score = (np.concatenate([np.asarray(c[i]) for c in doc["nominations"]])
                                  for i in range(3))
        row = np.arange(len(nom))
        input_x = x[0][row, nom]
        target = torch.from_numpy(model.events(gain[row, nom], b, known, (raw > 0.) & (raw <= local_cut)))
        cal_blocks = [r for r in blocks if r["source"] in cal]
        cal_cells = []
        for r in cal_blocks:
            s = r["sl"]
            p = r["bank"][np.arange(len(r["cv"])), nom[s]]
            # Illegal nominees are exact identity and never selected.
            cal_cells.append((legal[s], calibration_cell(r, p, cut)))
        for seed in SEEDS:
            net, artifact = fit_model(binding, held, seed, fit, cal, local_cut, input_x, target, groups, legal, beat, replay)
            p = PRIVATE/f"calibration_{held}_{seed}.json"
            if not p.exists():
                if replay:
                    raise ValueError("Replay may not choose a threshold")
                scores = dict(support=[predict(net, input_x[r["sl"]]) for r in cal_blocks],
                              gain=[gain_score[r["sl"]] for r in cal_blocks])
                ladder = {k: [trial(cal_blocks, scores[k], cal_cells, t) for t in grid]
                          for k, grid in model.THRESHOLDS.items()}
                selected = {k: model.select(v, model.THRESHOLDS[k]) for k, v in ladder.items()}
                run.once(p, run.stable(dict(registration=binding, checkpoint=artifact, held=held, seed=seed,
                    calibration=cal, fit=fit, ladder=ladder, selected=selected)))
            c = run.read(p)
            if (c["registration"], c["checkpoint"], c["fit"], c["calibration"]) != (binding, artifact, fit, cal):
                raise ValueError("Invalid calibration checkpoint")
            calibration.append(c)
            for r in blocks:
                if r["source"] != held:
                    continue
                s = r["sl"]
                path = PRIVATE/f"choices_{r['rec']}_{seed}.json"
                key = dict(registration=binding, calibration=run.artifact(p), nominee=run.artifact(nom_path),
                           record=r["rec"], seed=seed)
                if not path.exists():
                    if replay:
                        raise ValueError("Replay may not infer")
                    supp = predict(net, input_x[s])
                    gain_s = gain_score[s]
                    inv, counts, ok = r["inverse"], r["counts"], legal[s]
                    chosen = [model.pick(z, ok, inv, counts, -np.inf) for z in (supp, gain_s)]
                    for arm, z in (("support", supp), ("gain", gain_s)):
                        t = c["selected"][arm]
                        chosen.append(model.pick(z, ok, inv, counts, t if t is not None else np.inf))
                    same = np.bincount(inv[chosen[2]], minlength=len(counts))
                    chosen.append(model.pick(gain_s, ok, inv, same, -np.inf))
                    run.once(path, run.stable(dict(key, scores=[supp.tolist(), gain_s.tolist()], legal=ok.tolist(),
                        nom=nom[s].tolist(), actions=[np.where(q, nom[s], -1).tolist() for q in chosen])))
                held_cell = run.read(path)
                if any(held_cell[k] != v for k, v in key.items()):
                    raise ValueError("Frozen outer prediction changed")
                marks.append(run.artifact(path))
    if (len(marks), len(calibration)) != (72, 27):
        raise ValueError("Complete three-seed outer roster required")
    run.once(OUT/"freeze.json", run.stable(dict(registration=binding, calibration=[
        {k: v for k, v in r.items() if k != "ladder"} for r in calibration], predictions=marks,
        official_calibration_access=False, confirmation_access=False, all_outer_choices_frozen_before_score=True)))
    return marks, calibration


def score(binding, marks, calibration, blocks, cut, beat, replay):
    lookup = {b["rec"]: b for b in blocks}
    rows, quality = [], []
    for mark in marks:
        d = run.read(run.checked(mark))
        rec, seed = d["record"], d["seed"]
        b = lookup[rec]
        p = PRIVATE/f"score_{rec}_{seed}.json"
        if not p.exists():
            if replay:
                raise ValueError("Replay must not score missing records")
            paths = {CV: b["cv"], OLS: b["ols"]}
            qcounts = []
            for k, a in zip(ARMS, d["actions"], strict=True):
                a = np.asarray(a)
                on = np.flatnonzero(a >= 0)
                value = b["cv"].copy()
                value[on] = b["bank"][on, a[on]]
                paths[k] = value
                qcounts.append(np.bincount(b["inverse"][on], minlength=len(b["counts"])))
            np.testing.assert_array_equal(qcounts[0], qcounts[1])
            np.testing.assert_array_equal(qcounts[2], qcounts[4])
            row = prior.prior.prior.base.score_recording(paths, PAIRS, b["target"], b["valid"], b["inverse"],
                                                        rec, easy_cut=cut["easy"], hard_cut=cut["hard"])
            legal = np.asarray(d["legal"])
            nom = b["bank"][np.arange(len(legal)), np.asarray(d["nom"])]
            m = moments(b["cv"], nom, b["target"], b["valid"], cut["easy"])
            labels = (m["known"] & (m["diff"] > 0.) & (m["base"] > 0.),
                      m["easy"] & (m["easy_excess"] <= 0.))
            metrics = []
            w = b["weight"][legal]
            for y in labels:
                y = y[legal]
                metrics.append(dict(prevalence=float(np.average(y, weights=w)) if len(w) else None,
                    support_ap=float(average_precision_score(y, np.asarray(d["scores"][0])[legal], sample_weight=w))
                               if y.any() else None,
                    gain_ap=float(average_precision_score(y, np.asarray(d["scores"][1])[legal], sample_weight=w))
                            if y.any() else None))
            run.once(p, run.stable(dict(choice=mark, row=dict(row, source=b["source"], seed=seed), ap=metrics)))
        r = run.read(p)
        if r["choice"] != mark:
            raise ValueError("Scoring artifact does not bind the frozen policy")
        rows.append(r["row"])
        quality.append(dict(recording=rec, seed=seed, source=b["source"], ap=r["ap"]))
        if len(rows) % 12 == 0:
            beat(stage="score", completed=len(rows))
    summary = grouped_readout(rows, {b["rec"]: b["source"] for b in blocks}, SEEDS)
    run.once(OUT/"analysis.json", run.stable(dict(registration=binding, summary=summary, fold_selection=[
        {k: v for k, v in r.items() if k != "ladder"} for r in calibration],
        selected_folds={k: sum(r["selected"][k] is not None for r in calibration) for k in model.THRESHOLDS},
        ranking_quality=quality, fits=27, no_official_calibration=True, confirmation_access=False,
        strict_query_local=True, deployment=False)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding, identity, cut = register()

        def beat(**v):
            v.update(pid=os.getpid(), seconds=round(time.monotonic()-start, 1))
            run.kin.prior.source.atomic(PRIVATE/"heartbeat.json", v)
            if v.get("step", STEPS) == STEPS:
                print(v, flush=True)

        data = prior.prior.prepare(identity, beat)
        marks, cal = freeze(binding, data, cut, beat, a.replay)
        score(binding, marks, cal, data[2], cut, beat, a.replay)


if __name__ == "__main__":
    main()
