"""Replay the frozen margin model and localize optimistic harm estimates."""

from collections import Counter
from pathlib import Path
import platform
import sys

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_margin_assignment as prior

av, np, torch, study, base, eq, prev, old = (
    prior.av, prior.np, prior.torch, prior.study, prior.base, prior.eq, prior.prev, prior.old)
TERMS = ("all_positive_harm", "easy_positive_harm", "signed_easy")


def load(fold):
    ck = torch.load(study.checked(fold["fit"]["checkpoint"]), weights_only=True, map_location="cpu")
    if ck["identity"] != fold["identity"] or ck["step"] != prior.STEPS:
        raise ValueError("The prior margin model is not sealed")
    key = "supervised."
    state = {k[len(key):]: v for k, v in ck["net"].items() if k.startswith(key)}
    net = prior.model.Assignment(state["edit.0.weight"].shape[1]).eval()
    net.load_state_dict(state)
    return net


@torch.inference_mode()
def choice(net, d, m, cut):
    x, ctx, opts, row = prior.features(d, m, cut)
    unit = torch.from_numpy(m["unit"]).float()
    a, b = net(x, ctx)
    margins = prior.model.margins(a, b, opts, row, unit)
    logp = prior.model.slate_score(margins, b[:, 2], opts, row)
    ix = int(logp.argmax())
    path = av.slate.assemble(d["cv"], m["bank"], m["row"], m["col"], m["opts"][ix:ix+1])[0]
    return ix, path, float(logp[ix].exp()), -margins[ix, 2:].numpy()


def actual(d, m, y, v, cut, ix):
    a, b = prior.model.components(d["cv"], d["ols4"], m["bank"], y, v, cut[0], m["row"], m["col"], m["unit"])
    vals = (a, b, m["opts"][ix:ix+1], m["row"], m["unit"])
    return -prior.model.margins(*map(torch.from_numpy, vals))[0, 2:].numpy()


def row(net, d, seed, cut, truth, threshold):
    m = old.action.menu(d, seed, cut)
    if m is None:
        return None
    ix, p, score, pred = choice(net, d, m, cut)
    measured = actual(d, m, *truth, cut, ix)
    stats = prev.metrics(p, d, *truth, cut)
    return dict(pred=pred, actual=measured, released=threshold is not None and score >= threshold,
        eligible=eq.ceiling.eligible_additive(stats), supported=stats["risk"]["supported"],
        easy_defined=stats["risk"]["completion_easy_risk_upper"] is not None), p, score


def summarize(rows):
    out = Counter(views=len(rows), risk_pass=sum(r["supported"] for r in rows),
                  eligible=sum(r["eligible"] for r in rows), easy_undefined=sum(not r["easy_defined"] for r in rows))
    if not rows:
        return dict(out)
    for i, k in enumerate(TERMS):
        a, p = np.asarray([r["actual"][i] for r in rows]), np.asarray([r["pred"][i] for r in rows])
        residual = a-p
        out[k] = dict(violations=int((a > 0.).sum()), predicted_nonpositive=int((p <= 0.).sum()),
                      false_safe=int(((p <= 0.) & (a > 0.)).sum()),
                      median_residual=float(np.median(residual)), p90_residual=float(np.quantile(residual, .9)))
    return dict(out)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, data, src, roles, _, outer, _, _ = prior.register()
    path = prior.OUT/"summary.json"
    result = study.read(path)
    seal = study.read(study.checked(result["freeze"]))
    if seal["registration"] != key or seal["held_labels_used"] or result["reserved_outcomes_opened"]:
        raise ValueError("Require an immutable development run")
    truth = prev.load_targets(src, data, [n for n, d in data.items() if d["added"]])
    rows = {k: [] for k in ("calibration", "held")}
    city = {}
    for name, fold in seal["folds"].items():
        held, seed, cut = (fold["identity"][k] for k in ("held", "seed", "cut"))
        net = load(fold)
        threshold = fold["calibration"]["supervised"]["threshold"]
        src_fold = outer["folds"][name]
        cal = roles[held]["calibration"]
        names = [n for n in truth if data[n]["city"] == cal]
        raw, offsets = eq.predict(old.load_outer(src_fold), data, names)
        local_cal = []
        for n, lo, hi in zip(names, offsets[:-1], offsets[1:], strict=True):
            d = dict(data[n], bank=eq.expanded(data[n], raw[lo:hi]))
            if (r := row(net, d, seed, cut, truth[n], threshold)) is not None:
                rows["calibration"].append(r[0])
                local_cal.append(r[0])
        if sum(r["released"] for r in local_cal) != fold["calibration"]["supervised"]["accepted"]:
            raise ValueError("Calibration replay did not reproduce admissions")
        local_held = []
        with (np.load(study.checked(src_fold["forecast"]), allow_pickle=False) as raw,
              np.load(study.checked(fold["forecast"]), allow_pickle=False) as z):
            for k in ("names", "offsets"):
                np.testing.assert_array_equal(raw[k], z[k])
            for i, (n, lo, hi) in enumerate(zip(z["names"], z["offsets"][:-1], z["offsets"][1:], strict=True)):
                if data[n]["city"] != held:
                    raise ValueError("Wrong held city")
                d = dict(data[n], bank=eq.expanded(data[n], raw["raw_eq"][lo:hi]))
                if (r := row(net, d, seed, cut, truth[n], threshold)) is not None:
                    info, p, score = r
                    np.testing.assert_array_equal(p, z["supervised"][lo:hi])
                    np.testing.assert_array_equal(score, z["score_supervised"][i])
                    rows["held"].append(info)
                    local_held.append(info)
        city[name] = dict(calibration=summarize(local_cal), held=summarize(local_held))
    out = {k: dict(all=summarize(r), released=summarize([v for v in r if v["released"]])) for k, r in rows.items()}
    if (out["held"]["released"]["views"] != result["counts"]["supervised_released"]
            or out["held"]["released"]["eligible"] != result["counts"]["supervised_released_eligible"]):
        raise ValueError("Held admissions did not replay")
    payload = dict(input=study.artifact(path), code=study.artifact(Path(__file__)), terms=list(TERMS),
        result_source="fresh_run", metrics=out, by_fold=city, reserved_outcomes_opened=False,
        retrained=False, thresholds_changed=False)
    base.immutable(prior.OUT/"harm_margin_audit.json", payload)
    print(study.json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
