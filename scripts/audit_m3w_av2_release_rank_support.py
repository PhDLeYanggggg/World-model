"""Isolate ordering among genuinely supported development forecasts."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_release_rank as run


def main():
    run.torch.set_num_threads(4)
    run.torch.set_num_interop_threads(1)
    key, state, _, prior = run.register()
    parent = state[8]
    p = run.OUT/"summary.json"
    s = run.study.read(p)
    f = run.study.read(run.study.checked(s["freeze"]))
    if (f["registration"] != key or f["held_labels_used"] or f["reserved_outcomes_opened"]
            or s["independent_confirmation"] or set(f["folds"]) != set(parent["folds"])):
        raise ValueError("Require frozen development-only scores")
    held, cells, original, receipts = [], [], [], {}
    for name, ref in f["folds"].items():
        z = run.study.read(run.study.checked(ref))
        pf = parent["folds"][name]
        path = run.support.LOCAL/f"{name}.support.json"
        ann = run.study.read(path)
        if ann["parent"] != pf or z["identity"]["source"] != pf:
            raise ValueError("Wrong support receipt")
        raw = run.support.prior.prior.LOCAL/f"{name}.held.score.json"
        receipts[name] = dict(outcome=run.study.artifact(raw), support=run.study.artifact(path))
        rows = list(run.support.prior.role_rows(z, pf, name, "held"))
        for r, a in zip(rows, ann["cells"], strict=True):
            if r["clip"] != a["clip"] or r["city"] != z["identity"]["held"]:
                raise ValueError("Wrong support alignment")
            r.update(support=a["support"], harm=a["cause"] == "observed_positive_harm")
        city = z["identity"]["held"]
        original.append(run.gate.matched_rows(rows, run.ARMS, city))
        cell = run.gate.matched_rows([r for r in rows if r["support"]], run.ARMS, city)
        if cell is not None:
            cells.append(cell)
        held.extend(rows)
    def totals(groups):
        return {a: {k: sum(r["arms"][a][k] for r in groups) for k in run.gate.FIELDS} for a in run.ARMS}

    if (len(held) != s["views"] or sum(r["support"] for r in held) != prior["support_views"]
            or totals(original) != s["matched"]["counts"]):
        raise ValueError("Original matched roster did not replay")
    d = dict(input=run.study.artifact(p), code=run.study.artifact(Path(__file__)), receipt_hashes=receipts,
             matched_n=sum(r["n"] for r in cells), counts=totals(cells),
             paired_pp={f"{a}_minus_{b}": {k: run.av.interval(cells,
                 lambda r: 100*(r["arms"][a][k]-r["arms"][b][k])/r["n"]) for k in run.gate.FIELDS}
                 for a, b in (("strict_point", "actor_support"), ("strict_rank", "actor_support"),
                              ("strict_rank", "strict_point"))},
             known_support_is_outcome_only=True, used_for_selection=False,
             independent_confirmation=False, reserved_outcomes_opened=False, result_source="fresh_run")
    run.base.immutable(run.OUT/"supported_only_audit.json", d)
    print(run.study.json.dumps(d, indent=2))


if __name__ == "__main__":
    main()
