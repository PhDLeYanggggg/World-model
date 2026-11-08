"""Audit risk errors specifically on larger bridge actions with predicted gain."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_signed_bridge_risk as study

run, np, torch, prior, source, transfer = study.run, study.np, study.torch, study.prior, study.source, study.transfer


@torch.inference_mode()
def evaluate(d, p, g, head, correction, cut, old):
    cv, ols, bank = (p[k] for k in ("cv", "ols", "bank"))
    y, valid, rec = (d[k] for k in ("y", "valid", "rec"))
    s = p["scale"][:, None, None]
    features = torch.from_numpy(p["x"])
    costs = g(features).numpy()*s
    z = head(features[:, -3:]).numpy()
    truth = study.model.signed_target(prior.model.targets(cv, ols, bank[:, -3:], y, valid, cut[0], p["scale"]))
    t = (truth <= 0.).all(2)
    eligible = p["legal"][:, -3:] & (costs[:, -3:, :2].min(2) > 0.)
    options = {"original": costs, "direct": study.model.replace_bridge_risk(costs, z, p["scale"], np.zeros_like(correction)),
               "donor_calibrated": study.model.replace_bridge_risk(costs, z, p["scale"], correction),
               "oracle_bridge_risk": study.model.replace_bridge_risk(costs, truth, p["scale"], np.zeros_like(correction))}
    out = {}
    for k, score in options.items():
        passed = (study.model.signed_target(score[:, -3:]) <= 0.).all(2)
        row = dict(eligible=int(eligible.sum()), true_safe=int((eligible & t).sum()),
                   false_accept=int((eligible & passed & ~t).sum()), false_reject=int((eligible & ~passed & t).sum()),
                   true_accept=int((eligible & passed & t).sum()))
        c = prior.model.query_choices(score, p["legal"], d["query"], p["motion"])["joint"]
        at = c >= 0
        path = cv.copy()
        path[at] = bank[at, c[at]]
        stats = prior.capacity.policy(path, cv, cv, y, valid, rec, easy_cut=cut[0], hard_cut=cut[1])["recordings"]
        positives = 0
        for r, a in stats.items():
            ix = rec == r
            positives += a["supported"] and all(transfer.paired(b[ix], path[ix], y[ix], valid[ix])[1].sum() > 0. for b in (cv, ols))
        row.update(edits=int(at.sum()), bridge_edits=int((c >= bank.shape[1]-3).sum()),
                   risk_passes=sum(a["supported"] for a in stats.values()), dual_positive_passes=positives)
        if k in old["arms"] and any(row[v] != old["arms"][k][v] for v in ("edits", "risk_passes", "dual_positive_passes")):
            raise ValueError("Admission diagnosis did not replay frozen policy")
        out[k] = row
    return out


def checked(meta, name, cls, step):
    z = torch.load(run.checked(meta[name]), weights_only=True, map_location="cpu")
    if run.stable(z["identity"]) != meta["identity"] or z["step"] != step:
        raise ValueError("Wrong checkpoint lineage")
    net = cls()
    net.load_state_dict(z["net"])
    return net.eval()


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, frozen, _, gain = study.register()
    files = [study.OUT / n for n in ("analysis.json", "head_freeze.json")]
    a, heads = [run.read(p) for p in files]
    if a["registration"] != key or a["heads"] != run.sha(files[1]) or a["confirmation_access"]:
        raise ValueError("Require the bound signed-risk experiment")
    f = study.OUT / "admission_registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), parent=[run.artifact(p) for p in files],
        candidate_condition="positive predicted increment against both references", oracle_only_changes_three_bridges=True,
        no_training=True, confirmation_access=False)))
    nets = transfer.frozen_models(*frozen[:5])
    data = {k: prior.old.load_site(k) for k in source.SITES}
    rows = []
    for j, site in enumerate(source.SITES):
        d = data[site]
        cut = np.quantile(np.concatenate([v["cut_samples"] for k, v in data.items() if k != site]), [.25, .75])
        for i, seed in enumerate(prior.old.prior.SEEDS):
            p = prior.prepare(d, transfer.forecast(d["h"], [nets[seed][j]]))
            old = a["by_site"][site][i]
            base = checked(gain["heads"][f"{site}/{seed}"], "checkpoint", prior.model.BridgeValue, prior.STEPS)
            head = checked(heads["heads"][f"{site}/{seed}/full"], "artifact", study.model.SignedRisk, study.STEPS)
            rows.append(evaluate(d, p, base, head, np.asarray(old["correction"]), cut, old))
    summary = {k: {f2: sum(r[k][f2] for r in rows) for f2 in rows[0][k]} for k in rows[0]}
    run.once(study.OUT / "admission.json", run.stable(dict(registration=run.sha(f), summary=summary,
        calibration_is_not_certificate=True, no_training=True, confirmation_access=False)))


if __name__ == "__main__":
    main()
