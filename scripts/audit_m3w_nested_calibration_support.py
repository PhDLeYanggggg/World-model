"""Separate structural support, nominee coverage and threshold ranking."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_nested_recipient_calibration as job
from src.evaluation.m3w_recipient_release_capacity import evidence


run, np = job.run, job.np


def choose_known(gain, anchor, inverse, counts, weight):
    """Hindsight witness with one necessary safe-easy anchor, never forced counts."""
    nonharm = np.isfinite(gain) & (gain >= 0.)
    on = job.model.pick(gain*weight, nonharm, inverse, counts, -np.inf)
    if (on & anchor).any():
        return on
    ids = np.flatnonzero(anchor)
    if not len(ids):
        return None
    # Find the cheapest single swap needed to preserve a positive easy reference.
    last = {}
    for i in np.flatnonzero(on):
        g = int(inverse[i])
        if g not in last or gain[i]*weight[i] < gain[last[g]]*weight[last[g]]:
            last[g] = i
    used = np.bincount(inverse[on], minlength=len(counts))
    def cost(i):
        q = int(inverse[i])
        previous = gain[last[q]]*weight[last[q]] if used[q] >= counts[q] else 0.
        return gain[i]*weight[i]-previous
    i = max(ids, key=cost)
    q = int(inverse[i])
    if used[q] >= counts[q]:
        on[last[q]] = False
    on[i] = True
    return on


def inspect(block, choices, cut):
    b, bank = block, block["bank"]
    legal = np.any(bank != b["cv"][:, None], axis=(2, 3))
    ref = job.errors(b["cv"], b["target"], b["valid"])[0]
    known = b["valid"].any(1)
    easy = known & (ref > 0.) & (ref <= cut["easy"])
    allowed = b["counts"][b["inverse"]] > 0
    scores = np.stack([job.errors(p, b["target"], b["valid"])[0] for p in bank.swapaxes(0, 1)], axis=1)
    gain = np.where(legal & known[:, None] & allowed[:, None], ref[:, None]-scores, -np.inf)
    results = {}
    for arm, nom in choices.items():
        q = gain[np.arange(len(gain)), nom]
        anchor = easy & np.isfinite(q) & (q >= -.02*ref)
        on = choose_known(q, anchor, b["inverse"], b["counts"], b["weight"])
        v = None
        if on is not None:
            picked = np.flatnonzero(on)
            v = evidence(b["cv"], b["ols"], bank, b["target"], b["valid"], b["inverse"], None,
                         b["weight"], b["rec"], cut, picked, nom[picked])
            if not v["risk"]["supported"]:
                raise ValueError("Individually safe anchor should construct a feasible witness")
        results[arm] = dict(anchor_rows=int(anchor.sum()), witness=v)
    return dict(recording=b["rec"], source=b["source"], easy_rows=int(easy.sum()),
                easy_in_nonzero_quota=int((easy & allowed).sum()), arms=results)


def main():
    binding, identity, cut = job.register()
    freeze = run.read(job.OUT/"freeze.json")
    frontier = run.read(job.OUT/"frontier.json")
    if freeze["registration"] != binding or frontier["confirmation_access"]:
        raise ValueError("Require the frozen nested study")
    _, _, blocks = job.prior.prior.prepare(identity, lambda **_: None)
    rows = []
    for cal in (r for r in freeze["calibration"] if r["seed"] == job.SEEDS[0]):
        held = cal["held"]
        nom = run.read(job.PRIVATE/f"nomination_{held}.json")
        if nom["registration"] != binding or nom["cal"] != cal["calibration"] or held in nom["fit"]:
            raise ValueError("Unexpected nominee lineage")
        records = []
        for b, (a, _, _) in zip(blocks, nom["nominations"], strict=True):
            if b["source"] not in cal["calibration"]:
                continue
            legal = np.any(b["bank"] != b["cv"][:, None], axis=(2, 3))
            val = np.stack([job.errors(p, b["target"], b["valid"])[0]
                            for p in b["bank"].swapaxes(0, 1)], axis=1)
            best = np.where(legal, np.nan_to_num(val, nan=np.inf), np.inf).argmin(1)
            records.append(inspect(b, {"nominee": np.asarray(a), "full_bank": best}, cut))
        rows.append(dict(held=held, calibration=cal["calibration"], recordings=records))
    keys = ("nominee", "full_bank")
    run.once(job.OUT/"support_audit.json", run.stable(dict(frontier=run.artifact(job.OUT/"frontier.json"),
        rows=rows, source_folds=len(rows), folds_with_any_unsupported_recording={
            k: sum(any(r["arms"][k]["anchor_rows"] == 0 for r in f["recordings"]) for f in rows) for k in keys},
        folds_with_missing_easy_roster=sum(any(r["easy_rows"] == 0 for r in f["recordings"]) for f in rows),
        hindsight_risk_only=True, calibration_diagnostic_only=True, outer_evaluation_used=False,
        confirmation_access=False, thresholds_changed=False)))


if __name__ == "__main__":
    main()
