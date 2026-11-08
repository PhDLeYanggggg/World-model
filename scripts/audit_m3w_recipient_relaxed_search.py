"""Check predicted recipient feasibility beyond the finite price grid."""

from itertools import combinations
from pathlib import Path
import sys

from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_recipient_feasibility as job
from src.evaluation.m3w_recipient_release_capacity import evidence


run, np, torch = job.run, job.np, job.torch


def relaxed(gain, excess, legal, k):
    """Fractional feasibility is necessary; integer witnesses are checked."""
    rows = np.flatnonzero(legal)
    g, a = gain[rows], excess[rows]
    if k == 0:
        return "empty_unsupported", None
    if not 0 < k <= len(rows):
        raise ValueError("Illegal recording quota")
    scale = np.maximum(np.abs(a).max(0), 1e-9)
    a = a/scale
    tol = 1e-9*max(1, k)
    lower = np.partition(a, k-1, axis=0)[:k].sum(0)
    if np.any(lower > tol):
        return ("easy_mass_lower_bound" if lower[1] > tol else "all_mass_lower_bound"), None
    r = linprog(-g/max(float(np.abs(g).max()), 1e-9), A_ub=a.T, b_ub=np.zeros(2),
                A_eq=np.ones((1, len(rows))), b_eq=[k], bounds=(0., 1.), method="highs-ds")
    if r.status == 2:
        return "fractional_infeasible", None
    if not r.success:
        return "solver_unresolved", None
    fixed = np.flatnonzero(r.x >= 1.-1e-7)
    fractional = np.flatnonzero((r.x > 1e-7) & (r.x < 1.-1e-7))
    need = k-len(fixed)
    if not 0 <= need <= len(fractional) or len(fractional) > 12:
        return "fractional_only", None
    feasible = []
    for fill in combinations(fractional, need):
        picked = np.r_[fixed, np.array(fill, int)]
        if np.all(a[picked].sum(0) <= tol):
            feasible.append(picked)
    if not feasible:
        return "fractional_only", None
    return "integer_witness", rows[max(feasible, key=lambda ix: g[ix].sum())]


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    binding, identity, cut, parent = job.register()
    freeze = run.read(job.OUT/"freeze.json")
    result = run.read(job.OUT/"analysis.json")
    if freeze["registration"] != binding or result["registration"] != binding or result["confirmation_access"]:
        raise ValueError("Require the unchanged frozen held-source study")
    x, hx, _, _, legal, blocks = job.prepare(identity, lambda **_: None)
    blocks = {b["rec"]: b for b in blocks}
    before = job.previous_predictions(parent)
    rows = []
    for mark in freeze["predictions"]:
        choice = run.read(run.checked(mark))
        rec, seed = choice["recording"], choice["seed"]
        b = blocks[rec]
        s = b["sl"]
        state = torch.load(run.checked(choice["checkpoints"][1]), map_location="cpu", weights_only=True)
        if state["key"]["held"] != b["source"] or state["key"]["registration"] != binding:
            raise ValueError("Wrong source-excluded model")
        net = job.method.RecipientHead(hx[0].shape[1])
        net.load_state_dict(state["model"])
        ref, risk, _ = net.eval()(hx[0][s], hx[1][s])
        ref, risk = ref.numpy()*b["scale"][:, None], risk.numpy()*b["scale"][:, None]
        gain, a, e = risk.T
        excess = np.column_stack((a, e))-.02*ref
        k = int(b["counts"].sum())
        status, picked = relaxed(gain*b["weight"], excess, legal[s], k)
        c = np.asarray(choice["actions"][job.ARMS.index("support_record")])
        price_on = c >= 0
        predicted_price_pass = bool(k and (excess[price_on].sum(0) <= 0.).all())
        if predicted_price_pass and status in ("easy_mass_lower_bound", "all_mass_lower_bound", "fractional_infeasible"):
            raise ValueError("A valid price allocation contradicts the relaxation")
        witness = None
        if picked is not None:
            nom = job.prior.model.SupportMoments(x[0].shape[2])
            origin = before[rec, seed][1]
            state = torch.load(run.checked(origin["fits"][1]), map_location="cpu", weights_only=True)
            if state["identity"]["held"] != b["source"]:
                raise ValueError("Wrong path nominator")
            nom.load_state_dict(state["model"])
            _, costs = job.prior.predict(nom.eval(), b, x)
            candidates = np.any(b["bank"] != b["cv"][:, None], axis=(2, 3))
            a = job.method.nominate(costs, candidates)
            witness = evidence(b["cv"], b["ols"], b["bank"], b["target"], b["valid"], b["inverse"],
                               None, b["weight"], rec, cut, picked, a[picked])
        rows.append(dict(recording=rec, source=b["source"], seed=seed, status=status,
                         predicted_price_pass=predicted_price_pass, actual_witness=witness))
    states = sorted({r["status"] for r in rows})
    run.once(job.OUT/"search_audit.json", run.stable(dict(
        parent=run.artifact(job.OUT/"freeze.json"), rows=rows,
        counts={k: sum(r["status"] == k for r in rows) for k in states},
        predicted_price_passes=sum(r["predicted_price_pass"] for r in rows),
        actual_witness_passes=sum(r["actual_witness"] is not None and r["actual_witness"]["risk"]["supported"]
                                  for r in rows), same_recording_budget=True,
        confirmation_access=False, calibration_access=False, deployment=False)))


if __name__ == "__main__":
    main()
