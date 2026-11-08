"""Group-excluded recipient models, same path nominator and matched budgets."""

import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import platform
import sys
import time

if sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "4"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_matched_support_moments as prior
from scripts.audit_m3w_matched_support_recipients import optimize_paths
from src.evaluation.m3w_recipient_release_capacity import evidence
from src.world_model import m3w_recipient_feasibility as method


run, np, torch = prior.run, prior.np, prior.torch
OUT = prior.OUT.parent/"recipient_feasibility_v1"
PRIVATE = prior.PRIVATE.parent/OUT.name
SEEDS, STEPS, EVERY, BATCH = prior.SEEDS, 500, 100, 96
HEADS = ("minimum_moments", "support_supervised")
ARMS = ("prior_record", "recent_gain_record", "minimum_record", "support_record", "support_query")
CV, OLS = prior.prior.cal.BASE, prior.prior.cal.OLS
PAIRS = {f"{a}_vs_{'cv' if b == CV else b}": (b, a, CV)
         for a, b in ((OLS, CV), *((a, b) for a in ARMS for b in (CV, OLS)),
                      ("support_record", "prior_record"), ("support_record", "minimum_record"),
                      ("support_record", "recent_gain_record"), ("support_record", "support_query"))}


def register():
    parent, identity, cut = prior.register()
    freeze = run.read(prior.OUT/"freeze.json")
    if freeze["registration"] != parent or freeze["confirmation_access"] or freeze["calibration_access"]:
        raise ValueError("Require the unchanged held-source path nominator")
    p = OUT/"registration.json"
    run.once(p, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(method.__file__)),
        parent=run.artifact(prior.OUT/"freeze.json"), prior_recipient_certificate=run.artifact(prior.OUT/"recipient_audit.json"),
        primary="support_record_vs_prior_record_same_nominated_paths",
        design="attainable_risk_with_auxiliary_support_vs_same_model_without_auxiliary",
        steps=STEPS, batch_size=BATCH, checkpoint_every=EVERY, seeds=SEEDS, folds=9, lr=.001,
        original_risk_budget=.02, frozen_training_and_evaluation_roles=True,
        record_allocation_offline_diagnostic=True, query_quota_past_only=True,
        worker_processes=0, cpu_threads=4, calibration_access=False, confirmation_access=False)))
    return run.sha(p), identity, cut, freeze


def prepare(identity, beat):
    x, (base, gain, known, raw), blocks = prior.prepare(identity, beat)
    features, radius = x
    # Pool action descriptors symmetrically; no fixed candidate identities are labels.
    pooled = torch.cat((features.mean(1), features.amin(1), features.amax(1)), dim=1)
    g, debit, legal = method.attainable(gain, known, radius.numpy())
    truth = tuple(torch.from_numpy(a) for a in (base, g, debit, known))
    return x, (pooled, radius.amax(1)), truth, raw, legal, blocks


def fit_one(binding, held, arm, seed, x, truth, raw, groups, legal, beat, replay):
    p = PRIVATE/f"fit_{held}_{seed}_{arm}.pt"
    parts = [v[legal[v]] for k, v in groups.items() if k != held]
    ids = np.concatenate(parts)
    train_raw = raw[np.concatenate([v for k, v in groups.items() if k != held])]
    threshold = float(np.quantile(train_raw[np.isfinite(train_raw) & (train_raw > 0.)], .25))
    easy = torch.from_numpy(np.isfinite(raw) & (raw > 0.) & (raw <= threshold))
    key = dict(registration=binding, held=held, arm=arm, seed=seed, easy_cut=threshold,
               training_sources=[k for k in groups if k != held])
    torch.manual_seed(seed)
    net = method.RecipientHead(x[0].shape[1])
    initial = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in net.parameters())).hexdigest()
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    rng = torch.Generator().manual_seed(seed+7919)
    step, draw, trace = 0, "0"*64, []
    if p.exists():
        s = torch.load(p, map_location="cpu", weights_only=True)
        if s["key"] != key or s["initial"] != initial or not 0 < s["step"] <= STEPS:
            raise ValueError("Resume checkpoint differs")
        net.load_state_dict(s["model"])
        opt.load_state_dict(s["optimizer"])
        rng.set_state(s["rng"])
        step, draw, trace = s["step"], s["draw"], s["trace"]
    if replay and step != STEPS:
        raise ValueError("Replay must not fit")
    if not len(ids) or any(not len(v) for v in parts):
        raise ValueError("Missing fitting action support")
    for i in range(step, STEPS):
        ix = prior.prior.base.fit.draw_batch(parts, BATCH, rng)
        value = method.loss(net(x[0][ix], x[1][ix]), *(v[ix] for v in truth), easy[ix], arm == HEADS[1])
        if not torch.isfinite(value):
            raise FloatingPointError("Nonfinite recipient loss")
        opt.zero_grad(set_to_none=True)
        value.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        draw = hashlib.sha256(bytes.fromhex(draw)+ix.tobytes()).hexdigest()
        if (i+1) % EVERY == 0:
            trace.append(float(value.detach()))
            tmp = p.with_suffix(".pending")
            torch.save(dict(key=key, initial=initial, model=net.state_dict(), optimizer=opt.state_dict(),
                rng=rng.get_state(), step=i+1, draw=draw, trace=trace), tmp)
            tmp.replace(p)
            beat(stage="checkpoint", held=held, seed=seed, arm=arm, step=i+1, loss=trace[-1])
    return net.eval(), dict(checkpoint=run.artifact(p), key=key, initial=initial, draw=draw)


def previous_predictions(parent):
    result = {}
    for mark in parent["predictions"]:
        d = run.read(run.checked(mark))
        result[d["recording"], d["seed"]] = (mark, d)
    return result


@torch.inference_mode()
def choose(nets, b, x, head_x, before):
    h = b["sl"]
    state = torch.load(run.checked(before["fits"][1]), map_location="cpu", weights_only=True)
    if state["identity"]["held"] != b["source"] or state["identity"]["arm"] != "supervised":
        raise ValueError("Common nominator must exclude the evaluated group")
    path_net = prior.model.SupportMoments(x[0].shape[2])
    path_net.load_state_dict(state["model"])
    _, path_cost = prior.predict(path_net.eval(), b, x)
    legal = np.any(b["bank"] != b["cv"][:, None], axis=(2, 3))
    nom = method.nominate(path_cost, legal)
    ok = legal.any(1)
    n = len(ok)
    budget = int(b["counts"].sum())
    previous = np.asarray(before["decisions"][prior.ARMS.index("supervised_record")][0]) >= 0
    # Causal per-row score from the common head's original predicted recent-blend gain.
    simple = np.where(ok, path_cost[:, 0, 0]*b["weight"], -np.inf)
    recent = np.zeros(n, bool)
    recent[np.argsort(-simple, kind="stable")[:budget]] = True
    controls = [previous, recent]
    for i, net in enumerate(nets):
        ref, risk, _ = net(x=head_x[0][h], bound=head_x[1][h])
        ref, risk = ref.numpy()*b["scale"][:, None], risk.numpy()*b["scale"][:, None]
        args = (ref, risk[:, None], ok[:, None], b["inverse"], budget, b["weight"])
        controls.append(prior.model.allocate(*args)[1][0] >= 0)
        if i == 1:
            controls.append(prior.model.allocate(*args, query_counts=b["counts"])[1][0] >= 0)
    if any(c.sum() != budget or not ok[c].all() for c in controls):
        raise ValueError("Candidate eligibility or count differs")
    return [np.where(c, nom, -1).tolist() for c in controls]


def freeze(binding, data, parent, beat, replay):
    x, head_x, truth, raw, legal, blocks = data
    groups = prior.group_ids(blocks)
    before = previous_predictions(parent)
    fits, marks = [], []
    for g in groups:
        for seed in SEEDS:
            pair = [fit_one(binding, g, arm, seed, head_x, truth, raw, groups, legal, beat, replay) for arm in HEADS]
            (n0, a), (n1, b) = pair
            if (a["initial"], a["draw"], a["key"]["easy_cut"]) != (b["initial"], b["draw"], b["key"]["easy_cut"]):
                raise ValueError("Matched training draws or initialization changed")
            fits.extend((a, b))
            for row in blocks:
                if row["source"] != g:
                    continue
                path = PRIVATE/f"choice_{row['rec']}_{seed}.json"
                src, d = before[row["rec"], seed]
                key = dict(registration=binding, parent=src, recording=row["rec"], seed=seed,
                           checkpoints=[a["checkpoint"], b["checkpoint"]])
                if not path.exists():
                    if replay:
                        raise ValueError("Replay may not infer")
                    run.once(path, run.stable(dict(key, actions=choose((n0, n1), row, x, head_x, d))))
                frozen = run.read(path)
                if any(frozen[k] != v for k, v in key.items()):
                    raise ValueError("Held prediction changed")
                marks.append(run.artifact(path))
    if (len(fits), len(marks)) != (54, 72):
        raise ValueError("Incomplete three-seed roster")
    run.once(OUT/"freeze.json", run.stable(dict(registration=binding, fits=fits, predictions=marks,
        calibration_access=False, confirmation_access=False, all_choices_frozen_before_score=True)))
    return marks


def score(binding, marks, blocks, cut, beat, replay):
    lookup = {b["rec"]: b for b in blocks}
    rows, exact = [], []
    for mark in marks:
        d = run.read(run.checked(mark))
        rec, seed = d["recording"], d["seed"]
        b = lookup[rec]
        path = PRIVATE/f"score_{rec}_{seed}.json"
        if not path.exists():
            if replay:
                raise ValueError("Replay may not reopen scoring")
            paths = {CV: b["cv"], OLS: b["ols"]}
            witnesses = {}
            for arm, a in zip(ARMS, d["actions"], strict=True):
                a = np.asarray(a, int)
                on = np.flatnonzero(a >= 0)
                p = b["cv"].copy()
                p[on] = b["bank"][on, a[on]]
                if np.any(p != b["cv"], axis=(1, 2)).sum() != b["counts"].sum():
                    raise ValueError("Replacement count changed")
                if arm == "support_query" and not np.array_equal(
                        np.bincount(b["inverse"][on], minlength=len(b["counts"])), b["counts"]):
                    raise ValueError("Query count changed")
                paths[arm] = p
                best = optimize_paths(b["cv"], b["bank"], b["target"], b["valid"], on)
                v = evidence(b["cv"], b["ols"], b["bank"], b["target"], b["valid"], b["inverse"],
                             None, b["weight"], rec, cut, on, best)
                witnesses[arm] = dict(supported=v["risk"]["supported"],
                    both=bool(v["risk"]["supported"] and min(v["gain_mass"]) > 0.),
                    missing_easy=v["risk"]["completion_easy_risk_upper"] is None)
            row = prior.prior.base.score_recording(paths, PAIRS, b["target"], b["valid"], b["inverse"],
                                                   rec, easy_cut=cut["easy"], hard_cut=cut["hard"])
            run.once(path, run.stable(dict(choice=mark, row=dict(row, seed=seed, source=b["source"]),
                                          witnesses=witnesses)))
        r = run.read(path)
        if r["choice"] != mark or d["registration"] != binding:
            raise ValueError("Score does not bind frozen choices")
        rows.append(r["row"])
        exact.append(r["witnesses"])
        if len(rows) % 12 == 0:
            beat(stage="score", completed=len(rows))
    summary = prior.prior.base.grouped_readout(rows, {b["rec"]: b["source"] for b in blocks}, SEEDS)
    if any(summary["risk"][k]["switches"] != 95769 for k in ARMS):
        raise ValueError("Identical budgets required")
    bounds = {k: {v: sum(r[k][v] for r in exact) for v in ("supported", "both", "missing_easy")} for k in ARMS}
    run.once(OUT/"analysis.json", run.stable(dict(registration=binding, summary=summary, recipient_bound=bounds,
        fitted_models=54, candidate_paths_unchanged=True, all_arms_same_nominator=True, offline_record_allocation=True,
        calibration_access=False, confirmation_access=False, deployment=False)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding, identity, cut, parent = register()

        def beat(**v):
            v.update(pid=os.getpid(), seconds=round(time.monotonic()-start, 1))
            run.kin.prior.source.atomic(PRIVATE/"heartbeat.json", v)
            if v.get("step", STEPS) == STEPS:
                print(v, flush=True)

        data = prepare(identity, beat)
        marks = freeze(binding, data, parent, beat, a.replay)
        score(binding, marks, data[-1], cut, beat, a.replay)


if __name__ == "__main__":
    main()
