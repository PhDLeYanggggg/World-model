"""Held development attribution; never used to calibrate or release forecasts."""

from collections import Counter, defaultdict
from pathlib import Path
import platform
import sys

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_assignment_cost as run


np, torch, study = run.np, run.torch, run.study
TERMS = ("support", "risk", "utility", "pair")


def replace_terms(predicted, actual, terms):
    c, x, z = [a.copy() for a in predicted]
    ct, xt, zt = actual
    if "support" in terms:
        c[:, 0] = np.where(ct[:, 0] > 0., 50., -50.)
    if "risk" in terms:
        x[:, :, 2:5], c[:, 2] = xt[:, :, 2:5], ct[:, 2]
    if "utility" in terms:
        x[:, :, :2], c[:, 1] = xt[:, :, :2], ct[:, 1]
    return c, x, zt if "pair" in terms else z


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, data, src, _, _, _, control, _, _, parent = run.register()
    path = run.OUT/"summary.json"
    result = study.read(path)
    freeze = study.read(study.checked(result["freeze"]))
    if (freeze["registration"] != key or freeze["held_labels_used"] or freeze["reserved_outcomes_opened"]
            or result["independent_confirmation"] or result["reserved_outcomes_opened"]):
        raise ValueError("Require already scored development-only experiment")
    truth = run.prev.load_targets(src, data, [n for n, d in data.items() if d["added"]])
    variants = {"predicted": (), **{f"known_{k}": (k,) for k in TERMS},
                "known_additive_terms": TERMS[:3], "known_all_terms": TERMS}
    cities = defaultdict(Counter)
    for name, ref in freeze["folds"].items():
        f = study.read(study.checked(ref))
        receipt = study.read(run.LOCAL/f"{name}.score.json")
        if receipt["freeze"] != ref:
            raise ValueError("Wrong scored receipt")
        net = run.model.Heads(f["fit"]["widths"])
        ck = torch.load(study.checked(f["fit"]["checkpoint"]), weights_only=True, map_location="cpu")
        if ck["identity"] != f["identity"] or ck["step"] != run.STEPS:
            raise ValueError("Wrong learned head")
        net.load_state_dict(ck["net"])
        net.eval()
        held, seed, cut = (f["identity"][k] for k in ("held", "seed", "cut"))
        counts, k = cities[held], 0
        with np.load(study.checked(control["folds"][name]["forecast"]), allow_pickle=False) as a:
            for n, lo, hi in zip(a["names"], a["offsets"][:-1], a["offsets"][1:], strict=True):
                d = data[n]
                if d["city"] != held:
                    raise ValueError("Wrong city")
                if len(run.eq.radii(d, seed)) != 2:
                    continue
                q, r = f["rows"][k], receipt["rows"][k]
                k += 1
                p, _, opts, _, digest = run.prior.candidate(d, a["additive"][lo:hi])
                if not q["clip"] == r["clip"] == n or digest != q["hash"]:
                    raise ValueError("Wrong fixed bank")
                y, valid = truth[n]
                cx, ax, px, scale = run.model.inputs(d, p, opts, cut[0])
                pred = (net.context(torch.from_numpy(cx)).numpy(), net.action(torch.from_numpy(ax)).numpy(),
                        net.pair(torch.from_numpy(px)).numpy()[:, 0])
                target = run.model.targets(d, p, opts, y, valid, cut[0], scale)
                ok, support = run.prior.model.eligibility(d["cv"], d["ols4"], p, y, valid, cut[0], opts)
                counts["views"] += 1
                counts["any_support"] += int(support.any())
                counts["exact_capacity"] += int(ok.any())
                for v, terms in variants.items():
                    ix, _ = run.model.choose(*replace_terms(pred, target, terms), opts)
                    if v == "predicted" and (ix.tolist() != q["choice"] or
                            any(bool(e) != r["arms"][h]["eligible"] for h, e in zip(run.model.ARMS, ok[ix], strict=True))):
                        raise ValueError("Exact bank and frozen predictions must reproduce the scored arms")
                    for arm, option in zip(run.model.ARMS, ix, strict=True):
                        counts[f"{v}:{arm}"] += int(ok[option])
                        if v == "predicted" and ok.any() and not ok[option]:
                            counts[f"missed_opportunity:{arm}:support"] += int(not support[option])
                            counts[f"missed_opportunity:{arm}:other"] += int(support[option])
        if k != len(f["rows"]) or k != len(receipt["rows"]):
            raise ValueError("Wrong scored roster length")
    total = sum(cities.values(), Counter())
    if total["views"] != result["views"] or total["exact_capacity"] != parent["capacity"]["reassign_transport"]:
        raise ValueError("Fixed bank capacity or held roster changed")
    out = dict(source=study.artifact(path), result_source="fresh_run", outcome_replacement_diagnostic=True,
               original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False,
               used_for_selection=False, totals=total, by_city=cities)
    run.base.immutable(run.OUT/"component_attribution.json", out)
    print(study.json.dumps(total, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
