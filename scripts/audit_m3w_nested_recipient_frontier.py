"""Audit every inner-calibration cutoff and held ranking on the fixed nominations."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_nested_recipient_calibration as job


run, np, torch = job.run, job.np, job.torch


def exact_frontier(blocks, scores, choices):
    """Query top-k is fixed; thresholds can only take score-level prefixes."""
    cells, levels = [], []
    for b, score, (legal, cell) in zip(blocks, scores, choices, strict=True):
        on = job.model.pick(score, legal, b["inverse"], b["counts"], -np.inf)
        ix = np.flatnonzero(on)
        ix = ix[np.argsort(-score[ix], kind="stable")]
        m, offset, change, denominator = cell
        w = b["weight"]
        v = np.column_stack((m["ref_easy"], m["all_excess"], m["easy_excess"], m["easy_signed"],
                             w*m["utility"], w*change))
        cells.append((score[ix], np.vstack((np.zeros(6), v[ix].cumsum(0))), .02*m["ref_easy"].sum(),
                      offset, denominator, b["source"]))
        levels.append(score[ix])
    threshold = np.unique(np.concatenate(levels))
    if not len(threshold):
        return dict(levels=0, full_risk=0, qualifies=0, maximum_passes=0)
    passed = np.zeros(len(threshold), int)
    masses = {}
    for level, cum, easy_limit, offset, denom, source in cells:
        k = np.searchsorted(-level, -threshold, side="right")
        m = cum[k]
        passed += (m[:, 0] > 0.) & (m[:, 1] <= 0.) & (m[:, 2] <= 0.) & (m[:, 3] <= easy_limit)
        mass = masses.setdefault(source, [np.zeros((len(threshold), 2)), 0.])
        mass[0] += m[:, 4:]+[0., offset]
        mass[1] += denom
    mean_gain = np.mean([g/d for g, d in masses.values()], axis=0)
    risk = passed == len(blocks)
    return dict(levels=len(threshold), full_risk=int(risk.sum()),
                qualifies=int((risk & (mean_gain > 0.).all(1)).sum()), maximum_passes=int(passed.max()))


def rank_summary(quality):
    out = {}
    for j, label in enumerate(("beneficial_nomination", "safe_easy_nomination")):
        sources = sorted({r["source"] for r in quality if r["ap"][j]["support_ap"] is not None})
        vals = np.array([np.mean([r["ap"][j]["support_ap"]-r["ap"][j]["gain_ap"] for r in quality
                                 if r["source"] == g and r["ap"][j]["support_ap"] is not None])
                         for g in sources])
        draws = np.random.default_rng(73107).integers(len(vals), size=(3000, len(vals)))
        out[label] = dict(positive_event_groups=len(vals),
            defined_recording_seed_views=sum(r["ap"][j]["support_ap"] is not None for r in quality),
            source_mean_AP_difference=float(vals.mean()),
            interval95=np.quantile(vals[draws].mean(1), [.025, .975]).tolist())
    return out


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    binding, identity, cut = job.register()
    analysis = run.read(job.OUT/"analysis.json")
    freeze = run.read(job.OUT/"freeze.json")
    if (analysis["registration"] != binding or freeze["registration"] != binding
            or analysis["confirmation_access"] or not freeze["all_outer_choices_frozen_before_score"]):
        raise ValueError("Require frozen inner and outer evidence")
    x, _, blocks = job.prior.prior.prepare(identity, lambda **_: None)
    rows = []
    for cal in freeze["calibration"]:
        held = cal["held"]
        if held in cal["calibration"] or set(cal["calibration"]) & set(cal["fit"]):
            raise ValueError("Calibration role overlaps fitting or held")
        fit = torch.load(run.checked(cal["checkpoint"]), map_location="cpu", weights_only=True)
        if fit["key"]["calibration"] != cal["calibration"] or fit["key"]["held"] != held:
            raise ValueError("Wrong calibrated checkpoint")
        nom = run.read(job.PRIVATE/f"nomination_{held}.json")
        if (nom["held"], nom["fit"], nom["cal"], nom["registration"]) != (held, cal["fit"], cal["calibration"], binding):
            raise ValueError("Nominee must exclude inner calibration")
        net = job.model.SupportRanker(x[0].shape[2])
        net.load_state_dict(fit["model"])
        cal_blocks, scores, choices = [], {"support": [], "gain": []}, []
        for b, (a, legal, g) in zip(blocks, nom["nominations"], strict=True):
            if b["source"] not in cal["calibration"]:
                continue
            i = np.arange(len(a))
            a = np.asarray(a)
            value = b["bank"][i, a]
            scores["gain"].append(np.asarray(g))
            scores["support"].append(job.predict(net.eval(), x[0][b["sl"]][i, a]))
            cal_blocks.append(b)
            choices.append((np.asarray(legal), job.calibration_cell(b, value, cut)))
        exact = {k: exact_frontier(cal_blocks, s, choices) for k, s in scores.items()}
        rows.append(dict(held=held, seed=cal["seed"], calibration_views=len(cal_blocks), exact=exact))
    run.once(job.OUT/"frontier.json", run.stable(dict(analysis=run.artifact(job.OUT/"analysis.json"), rows=rows,
        summary={k: dict(folds_with_qualifying_level=sum(r["exact"][k]["qualifies"] > 0 for r in rows),
                         folds_with_any_full_risk=sum(r["exact"][k]["full_risk"] > 0 for r in rows),
                         evaluated_levels=sum(r["exact"][k]["levels"] for r in rows)) for k in job.model.THRESHOLDS},
        held_ranking=rank_summary(analysis["ranking_quality"]), diagnostic_only=True,
        official_calibration_access=False, confirmation_access=False, thresholds_changed=False)))


if __name__ == "__main__":
    main()
