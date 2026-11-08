"""Strictly query-local selective policies using unchanged frozen risk heads."""

import fcntl
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_recipient_feasibility as job


run, np, torch = job.run, job.np, job.torch
OUT = job.OUT.parent/"recipient_query_fallback_v1"
PRIVATE = job.PRIVATE.parent/OUT.name
ARMS = ("recipient_rank", "path_rank")
PAIRS = {"recipient_vs_path": ("path_rank", "recipient_rank", job.CV),
         "recipient_vs_cv": (job.CV, "recipient_rank", job.CV),
         "recipient_vs_ols": (job.OLS, "recipient_rank", job.CV),
         "path_vs_cv": (job.CV, "path_rank", job.CV)}


def local_choices(inverse, counts, legal, ref, cost, recipient_gain):
    """No query can affect eligibility or ranking in another query."""
    ids = np.arange(len(inverse))
    eligible = legal & (cost[:, 0] > 0.) & (ref > 0.).all(1) & (cost[:, 1:] <= .02*ref).all(1)
    result = []
    for value in (recipient_gain, cost[:, 0]):
        on = np.zeros(len(ids), bool)
        order = np.lexsort((ids, -value, ~eligible, inverse))
        sizes = np.bincount(inverse, minlength=len(counts))
        rank = ids-np.repeat(np.r_[0, np.cumsum(sizes)[:-1]], sizes)
        take = order[(rank < counts[inverse[order]]) & eligible[order]]
        on[take] = True
        result.append(on)
    if not np.array_equal(np.bincount(inverse[result[0]], minlength=len(counts)),
                          np.bincount(inverse[result[1]], minlength=len(counts))):
        raise ValueError("Comparison must preserve query action counts")
    return result


def register():
    binding, identity, cut, parent = job.register()
    audit = run.read(job.OUT/"search_audit.json")
    if audit["counts"] != {"easy_mass_lower_bound": 69, "empty_unsupported": 3} or audit["confirmation_access"]:
        raise ValueError("Require the matched-budget infeasibility result")
    p = OUT/"registration.json"
    run.once(p, run.stable(dict(code=run.artifact(Path(__file__)), parent=run.artifact(job.OUT/"freeze.json"),
        diagnostic=run.artifact(job.OUT/"search_audit.json"), model_binding=binding,
        scope="post_matched_failure_fitting_only_no_threshold_selection",
        constraints="each nominated path individually passes both predicted 2 percent inequalities",
        controls="same per-query count and path nomination, attainable vs nominated gain order",
        calibration_access=False, confirmation_access=False, deployment=False)))
    return run.sha(p), identity, cut, parent


@torch.inference_mode()
def infer(binding, data, parent):
    x, hx, _, _, legal, blocks = data
    lookup = {b["rec"]: b for b in blocks}
    previous = job.previous_predictions(parent)
    marks = []
    for r in run.read(job.OUT/"freeze.json")["predictions"]:
        f = run.read(run.checked(r))
        rec, seed = f["recording"], f["seed"]
        p = PRIVATE/f"choices_{rec}_{seed}.json"
        b = lookup[rec]
        if not p.exists():
            sl = b["sl"]
            ps = torch.load(run.checked(previous[rec, seed][1]["fits"][1]), map_location="cpu", weights_only=True)
            rs = torch.load(run.checked(f["checkpoints"][1]), map_location="cpu", weights_only=True)
            if ps["identity"]["held"] != b["source"] or rs["key"]["held"] != b["source"]:
                raise ValueError("Held groups must be excluded from both fits")
            path_net = job.prior.model.SupportMoments(x[0].shape[2])
            path_net.load_state_dict(ps["model"])
            ref, cost = job.prior.predict(path_net.eval(), b, x)
            rec_net = job.method.RecipientHead(hx[0].shape[1])
            rec_net.load_state_dict(rs["model"])
            _, attain, _ = rec_net.eval()(hx[0][sl], hx[1][sl])
            valid = np.any(b["bank"] != b["cv"][:, None], axis=(2, 3))
            nom = job.method.nominate(cost, valid)
            row = np.arange(len(nom))
            choices = local_choices(b["inverse"], b["counts"], legal[sl], ref, cost[row, nom],
                                    attain[:, 0].numpy()*b["scale"])
            run.once(p, run.stable(dict(registration=binding, parent=r, recording=rec, seed=seed,
                actions=[np.where(on, nom, -1).tolist() for on in choices])))
        d = run.read(p)
        if d["registration"] != binding or d["parent"] != r:
            raise ValueError("Stale local choices")
        marks.append(run.artifact(p))
    run.once(OUT/"freeze.json", run.stable(dict(registration=binding, choices=marks,
        complete_before_scoring=True, no_future_labels_used_in_allocation=True)))
    return marks


def score(binding, blocks, cut, marks):
    lookup = {b["rec"]: b for b in blocks}
    rows = []
    for mark in marks:
        d = run.read(run.checked(mark))
        rec, seed = d["recording"], d["seed"]
        b = lookup[rec]
        p = PRIVATE/f"score_{rec}_{seed}.json"
        if not p.exists():
            paths = {job.CV: b["cv"], job.OLS: b["ols"]}
            totals = []
            for arm, a in zip(ARMS, d["actions"], strict=True):
                a = np.asarray(a, int)
                on = np.flatnonzero(a >= 0)
                path = b["cv"].copy()
                path[on] = b["bank"][on, a[on]]
                counts = np.bincount(b["inverse"][on], minlength=len(b["counts"]))
                if (counts > b["counts"]).any():
                    raise ValueError("Local action exceeds past-only quota")
                totals.append(counts)
                paths[arm] = path
            np.testing.assert_array_equal(*totals)
            row = job.prior.prior.base.score_recording(paths, PAIRS, b["target"], b["valid"], b["inverse"],
                                                        rec, easy_cut=cut["easy"], hard_cut=cut["hard"])
            run.once(p, run.stable(dict(parent=mark, row=dict(row, seed=seed, source=b["source"]))))
        r = run.read(p)
        if r["parent"] != mark or d["registration"] != binding:
            raise ValueError("Choice and score mismatch")
        rows.append(r["row"])
    summary = job.prior.prior.base.grouped_readout(rows, {b["rec"]: b["source"] for b in blocks}, job.SEEDS)
    risk = summary["risk"]
    if risk[ARMS[0]]["switches"] != risk[ARMS[1]]["switches"]:
        raise ValueError("The same-count control must have actual matching moves")
    run.once(OUT/"analysis.json", run.stable(dict(registration=binding, summary=summary,
        query_local_only=True, no_new_fits=True, empty_rows_are_not_risk_passes=True,
        calibration_access=False, confirmation_access=False, deployment=False)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding, identity, cut, parent = register()
        data = job.prepare(identity, lambda **v: None)
        marks = infer(binding, data, parent)
        score(binding, data[-1], cut, marks)


if __name__ == "__main__":
    main()
