"""Locate the observed and missing-label contributions to the transfer gap."""

from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_standalone_hindcast as job


run, cal, np = job.run, job.cal, job.np


def fitting_rows(identity, cut, arm):
    train = job.base.train_source
    data = train.fitting_only(identity, lambda **v: None)
    admission = run.read(train.ROOT/run.read(train.source.CONFIG)["admission"])
    support = run.read(train.bank.old.HOME/"movingdronecrowd_intake_v1/analysis.json")
    sizes = {r["clip"]: r["window_support"]["obs8_pred12_native_stride1"]["history_complete_agent_instances"]
             for r in support["recordings"]}
    lo = 0
    rows = []
    for rec in data["recordings"]:
        entry = admission["recordings"][rec]
        hi = lo+sizes[entry["clip"]]
        h = data["geometry"][lo:hi, :16].reshape(-1, 8, 2)
        cv = data["geometry"][lo:hi, 332:356].reshape(-1, 12, 2).astype(float)
        ols = job.base.motion.slope(h[:, -4:])[:, None]*np.arange(1., 13.)[None, :, None]
        action = job.method.menu(job.method.backtests(h), np.any(cv != ols, axis=(1, 2)))[arm]
        path = job.base.motion.make_path(cv, ols, action)
        m = job.base.score_recording({cal.BASE: cv, job.ARM: path}, {}, data["target"][lo:hi],
            data["valid"][lo:hi], np.arange(hi-lo), rec, easy_cut=cut["easy"], hard_cut=cut["hard"])
        rows.append(m["metrics"][job.ARM]["risk"])
        lo = hi
    if lo != len(data["ids"]):
        raise ValueError("Original fitting rows must align")
    return rows


def summarize(rows):
    risk_fields = ("known_all_risk", "known_easy_risk", "completion_all_risk_upper",
                   "completion_easy_risk_upper", "known_easy_net_degradation")
    return dict(recordings=len(rows), supported=sum(r["supported"] for r in rows),
        observed_only_violations=sum(any(r[k] is not None and r[k] > .02 for k in
            ("known_all_risk", "known_easy_risk", "known_easy_net_degradation")) for r in rows),
        selected=sum(r["selected"] for r in rows), unknown_selected=sum(r["unknown_selected"] for r in rows),
        risk={k: dict(undefined=sum(r[k] is None for r in rows),
                      over_2=sum(r[k] is not None and r[k] > .02 for r in rows),
                      worst=max((r[k] for r in rows if r[k] is not None), default=None)) for k in risk_fields})


def main():
    binding, identity, cut = job.register()
    fit = run.read(job.OUT/"fitting.json")
    freeze = run.read(job.OUT/"freeze.json")
    analysis = run.read(job.OUT/"analysis.json")
    if {x["registration"] for x in (fit, freeze, analysis)} != {binding} or fit["chosen"] != freeze["chosen"]:
        raise ValueError("Only the frozen diagnostic can be audited")
    _, _, roster, _, _ = job.target_context()
    arms = (job.ARM, job.MATCH, job.FIXED, cal.OLS)
    risk = {k: [] for k in arms}
    passes = defaultdict(list)
    for cell in freeze["clips"]:
        rec = cell["recording"]
        scored = run.read(job.PRIVATE/f"score_{rec}.json")
        if cell["choices"] != scored["choices"]:
            raise ValueError("Score must bind frozen choices")
        for k in arms:
            risk[k].append(scored["result"]["metrics"][k]["risk"])
        passes[roster[rec]].append(int(risk[job.ARM][-1]["supported"])-int(risk[job.MATCH][-1]["supported"]))
    values = np.array([np.mean(passes[g]) for g in sorted(passes)])
    draw = np.random.default_rng(73107).integers(len(values), size=(3000, len(values)))
    source = summarize(fitting_rows(identity, cut, fit["chosen"]))
    result = dict(
        analysis=run.artifact(job.OUT/"analysis.json"),
        fitting=source, opened_calibration={k: summarize(risk[k]) for k in arms},
        positive_matched_pass_groups=int((values > 0).sum()), paired_pass_fraction=float(values.mean()),
        paired_pass_fraction_interval95=np.quantile(values[draw].mean(1), [.025, .975]).tolist(),
        bootstrap_draws=3000, independent_confirmation=False, new_calibration_labels_read=False)
    if source["supported"] != sum(x["passed"] for x in fit["scores"][fit["chosen"]].values()):
        raise ValueError("Fitting safety must reproduce")
    run.once(job.OUT/"risk_gap.json", run.stable(result))


if __name__ == "__main__":
    main()
