"""Decompose frozen selected-risk error into harm and reference mass."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_matched_support_moments as job


run, np, torch = job.run, job.np, job.torch
ARMS = ("marginal_record", "supervised_record")


def true_mass(b, action, cut):
    on = action >= 0
    row = np.flatnonzero(on)
    cv, y, valid = (b[k][on] for k in ("cv", "target", "valid"))
    path = b["bank"][row, action[on]]
    e0 = job.errors(cv, y, valid)[0]
    e1 = job.errors(path, y, valid)[0]
    known = valid.any(1)
    easy = known & (e0 > 0.) & (e0 <= cut)
    harm = np.maximum(e1-e0, 0.)
    unknown = np.linalg.norm(path[~known]-cv[~known], axis=-1).max(1).sum()
    return (np.array([e0[known].sum(), e0[easy].sum()]),
            np.array([harm[known].sum()+unknown, harm[easy].sum()+unknown]))


def ratio(h, r):
    return [float(a/b) if b > 0. else None for a, b in zip(h, r, strict=True)]


def supported(r):
    return all(v is not None and v <= .02 for v in r)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    binding, identity, cut = job.register()
    result = run.read(job.OUT/"analysis.json")
    freeze = run.read(job.OUT/"freeze.json")
    if result["registration"] != binding or freeze["registration"] != binding or result["confirmation_access"]:
        raise ValueError("Only the frozen source diagnostic may be decomposed")
    x, _, blocks = job.prepare(identity, lambda **_: None)
    by_rec = {b["rec"]: b for b in blocks}
    rows = []
    for mark in freeze["predictions"]:
        d = run.read(run.checked(mark))
        b = by_rec[d["recording"]]
        scored = run.read(job.PRIVATE/f"score_{d['recording']}_{d['seed']}.json")
        if scored["forecast"] != mark:
            raise ValueError("Prediction and score alignment changed")
        cells = {}
        for arm, receipt in zip(ARMS, d["fits"], strict=True):
            action, recorded = d["decisions"][job.ARMS.index(arm)]
            action = np.asarray(action)
            on = action >= 0
            state = torch.load(run.checked(receipt), map_location="cpu", weights_only=True)
            if state["identity"]["held"] != b["source"] or state["identity"]["registration"] != binding:
                raise ValueError("Wrong held fold")
            net = job.model.SupportMoments(x[0].shape[2])
            net.load_state_dict(state["model"])
            ref, moment = job.predict(net.eval(), b, x)
            rp, hp = ref[on].sum(0), moment[np.flatnonzero(on), action[on], 1:].sum(0)
            rt, ht = true_mass(b, action, cut["easy"])
            predicted, actual = ratio(hp, rp), ratio(ht, rt)
            risk = scored["row"]["metrics"][arm]["risk"]
            for a, z in zip((*predicted, *actual),
                            (*recorded, risk["completion_all_risk_upper"], risk["completion_easy_risk_upper"]),
                            strict=True):
                if (a is None) != (z is None) or (a is not None and not np.isclose(a, z, atol=1e-10, rtol=1e-7)):
                    raise ValueError("Mass decomposition must reproduce original ratios")
            cells[arm] = dict(predicted=predicted, actual=actual, signed_gate=risk["known_easy_net_degradation"],
                predicted_harm_actual_ref=ratio(hp, rt), actual_harm_predicted_ref=ratio(ht, rp),
                predicted_harm=hp.tolist(), predicted_ref=rp.tolist(), actual_harm=ht.tolist(), actual_ref=rt.tolist())
        rows.append(dict(recording=d["recording"], seed=d["seed"], source=b["source"], cells=cells))
    summary = {}
    for arm in ARMS:
        cells = [r["cells"][arm] for r in rows]
        keys = ("predicted", "actual", "predicted_harm_actual_ref", "actual_harm_predicted_ref")
        eligible = [c for c in cells if all(v is not None for v in c["actual"])]
        summary[arm] = dict(risk_passes_all_views={k: sum(supported(c[k]) for c in cells) for k in keys},
            views_with_actual_denominators=len(eligible),
            risk_passes_common_support={k: sum(supported(c[k]) for c in eligible) for k in keys},
            reference_overestimates=[sum(c["predicted_ref"][j] > c["actual_ref"][j] for c in cells) for j in range(2)],
            harm_underestimates=[sum(c["predicted_harm"][j] < c["actual_harm"][j] for c in cells) for j in range(2)])
    run.once(job.OUT/"mass_audit.json", run.stable(dict(analysis=run.artifact(job.OUT/"analysis.json"),
        scope="diagnostic_fixed_actions_swap_one_risk_component_not_a_policy",
        independent_confirmation=False, rows=rows, summary=summary)))


if __name__ == "__main__":
    main()
