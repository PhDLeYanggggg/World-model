"""Fit a strict release ranker without changing AV2 forecasts or risk rules."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_observed_harm as prior
from scripts import run_m3w_av2_separate_support_gate as gate
from src.world_model import m3w_av2_release_rank as model

np, torch, study, base, av = prior.np, prior.torch, prior.study, prior.base, prior.av
old, support, rel, timing, bank = prior.old, prior.support, prior.rel, prior.timing, prior.bank
OUT = prior.OUT.parent/"av2_release_rank_v1"
LOCAL = prior.LOCAL.parent/"av2_release_rank_v1"
FITS = {"strict_point": 0., "strict_rank": 1.}
ARMS = (*prior.ARMS, *FITS)
STEPS, SAVE, PAIRS, RIVALS = 400, 50, 8, 4


def register():
    state = prior.register()
    p = prior.OUT/"summary.json"
    s = study.read(p)
    hf = study.read(study.checked(s["freeze"]))
    audit = prior.OUT.parent/"av2_release_transport_v1"/"transport_audit.json"
    a = study.read(audit)
    if (hf["registration"] != state[0] or hf["held_labels_used"] or hf["calibration_labels_used"]
            or s["reserved_outcomes_opened"] or s["independent_confirmation"]
            or a["used_for_selection"] or a["reserved_outcomes_opened"]):
        raise ValueError("Require frozen development projection scores")
    r = OUT/"registration.json"
    base.immutable(r, dict(code=[study.artifact(Path(p)) for p in (__file__, model.__file__)],
        parent=study.artifact(p), motivation=study.artifact(audit), arms=ARMS, rank_weights=FITS,
        seeds=rel.eq.SEEDS, steps=STEPS, save_every=SAVE, positive_pairs_per_donor=PAIRS, rivals=RIVALS,
        threads=4, workers=0,
        target="exact joint qualification: edited-easy support, original risk and both CV/OLS4 floors",
        batch="equal donor weights; sample source queries before options; rivals from different scenarios in same donor",
        hard_negatives="half of rival slots prefer genuinely supported but not qualified, if available",
        ranking="BCE in both arms; rank arm adds softplus margin 1 against highest-scored sampled rival",
        primary="paired fixed top five per held fold versus frozen actor-support and matched pointwise fit",
        calibration="same nonempty all-qualified threshold prefix as the parent",
        forecasts_frozen=True, original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(r), state, hf, s


def examples(data, truth, role, inner, refs, frozen):
    held, cal, donors, seed = (frozen["identity"][k] for k in ("held", "calibration", "donors", "seed"))
    nested = timing.old.donor_forecasts(inner, data, held, role, seed)
    if set(nested) != set(truth) or {data[n]["city"] for n in truth} != set(donors):
        raise ValueError("Wrong donor query roster")
    groups, labels = {}, {}
    for city in donors:
        lineage = frozen["lineage"][city]
        ref = refs["references"][city]
        eq = inner["models"][f"{role['inner'][city]}_{seed}"]
        training = set(donors)-{city}
        if (set(lineage["training"]) != training or {city, held, cal} & training
                or lineage["target_city"] != city or lineage["action"] != ref["checkpoint"]
                or lineage["eqmotion"] != eq["forecast"] or set(eq["fit"]["identity"]["donors"]) != training):
            raise ValueError("Overlapping producer lineage")
        action, cut = timing.rel.pair.inner_action(ref, refs["identity"], city)
        net, rows, status = None, [], []
        for n, (y, v) in truth.items():
            d = data[n]
            if d["city"] != city:
                continue
            a = timing.incumbent(d, seed, cut, nested[n], action, action_cut=cut)
            if a is None or np.any(a != d["cv"], axis=(1, 2)).sum() != 2:
                continue
            p, _, opts, _, _ = bank.candidate(d, a)
            actors = np.arange(len(p))
            if net is None:
                width = old.proj.inputs(d, p[:, 0], actors)[0].shape[1]
                net = prior.load_direction(lineage["direction"], lineage["training"],
                                           frozen["identity"]["registration"], seed, width)
            paths = np.stack([rel.base_positions(net, d, p[:, k], actors) for k in range(2)], axis=1)
            np.testing.assert_allclose(np.linalg.norm(paths-d["cv"][:, None], axis=-1),
                                       np.linalg.norm(p-d["cv"][:, None], axis=-1), rtol=1e-8, atol=1e-8)
            x, _ = old.model.inputs(d, paths, opts, cut[0])
            qualified, known = bank.model.eligibility(d["cv"], d["ols4"], paths, y, v, cut[0], opts)
            rows.append(x)
            status.append(np.column_stack((qualified, known)))
        if not rows:
            raise ValueError("No source-excluded donor forecasts")
        groups[city], labels[city] = rows, status
    if sum(len(q) for xs in groups.values() for q in xs) != frozen["fit"]["rows"]:
        raise ValueError("All-legal option roster changed")
    totals = Counter()
    for city in groups:
        yy = labels[city]
        totals.update(queries=len(yy), options=sum(len(q) for q in yy),
                      positive_queries=sum(bool(q[:, 0].any()) for q in yy),
                      qualified=sum(int(q[:, 0].sum()) for q in yy),
                      supported_fail=sum(int((q[:, 1] & ~q[:, 0]).sum()) for q in yy))
    return {city: (groups[city], model.label_pools(labels[city])) for city in sorted(groups)}, dict(totals)


def fit(ident, name, groups):
    torch.manual_seed(ident["seed"])
    rng = np.random.default_rng(ident["seed"]+1871)
    net = model.head(next(iter(groups.values()))[0][0].shape[-1])
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    p = LOCAL/f"{name}.pt"
    start, trace = 0, []
    if p.exists():
        ck = torch.load(p, weights_only=True, map_location="cpu")
        if ck["identity"] != ident or ck["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Wrong rank resume identity")
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["optimizer"])
        rng.bit_generator.state = ck["rng"]
        start, trace = ck["step"], ck["trace"]
    for step in range(start+1, STEPS+1):
        batch = np.asarray([[xs[i][j], *(xs[k][v] for k, v in ns)]
                            for xs, pools in groups.values()
                            for (i, j), ns in model.sample(pools, rng, PAIRS, RIVALS)])
        z = net(torch.from_numpy(batch)).squeeze(-1)
        loss, bce, rank = model.loss(z[:, 0], z[:, 1:], FITS[ident["arm"]])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % SAVE == 0:
            trace.append([float(bce.detach()), float(rank.detach())])
            tmp = p.with_suffix(".pending")
            torch.save(dict(identity=ident, step=step, net=net.state_dict(), optimizer=opt.state_dict(),
                            rng=rng.bit_generator.state, trace=trace), tmp)
            tmp.replace(p)
            beat("fit", fold=name, step=step, total=STEPS)
    return net.eval(), dict(checkpoint=study.artifact(p), trace=trace)


def beat(phase, **kw):
    r = dict(phase=phase, pid=os.getpid(), timestamp=time.time(), **kw)
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(r)+"\n")
    p.replace(LOCAL/"heartbeat.json")
    print(r, flush=True)


@torch.inference_mode()
def scores(nets, d, path, ix, cut):
    x, _ = old.model.inputs(d, np.repeat(path[:, None], 2, axis=1), ix[None], cut)
    return {a: float(m(torch.from_numpy(x))[0, 0]) for a, (m, _) in nets.items()}


def freeze(key, state, hf):
    _, data, src, roles, inner, outer, control, refs, parent, _, nf, _ = state
    done = {}
    for name, ref in parent["folds"].items():
        f = study.read(study.checked(ref))
        donor_ref = nf["folds"][name]
        donor = study.read(study.checked(donor_ref))
        h = study.read(study.checked(hf["folds"][name]))
        held, cal, donors, seed = (f["identity"][k] for k in ("held", "calibration", "donors", "seed"))
        if (len({held, cal, *donors}) != 6 or donor["identity"]["parent"] != ref
                or h["identity"]["source"] != ref or roles[held]["donors"] != donors
                or roles[held]["calibration"] != cal):
            raise ValueError("Wrong fold roles")
        ident = dict(registration=key, source=ref, donor=donor_ref, prior=hf["folds"][name],
                     held=held, calibration=cal, donors=donors, seed=seed)
        p = LOCAL/f"{name}.json"
        if not p.exists():
            beat("donors", fold=name)
            y = old.prev.load_targets(src, data, [n for n, d in data.items() if d["city"] in donors])
            groups, training = examples(data, y, roles[held], inner, refs["folds"][name], donor)
            nets = {a: fit(dict(ident, arm=a), f"{name}_{a}", groups) for a in FITS}
            rows = {}
            for side in ("calibration", "held"):
                rows[side] = []
                paths = support.frozen_paths(data, f, control["folds"][name], outer["folds"][name], side)
                for (n, d, ix, path), q in zip(paths, h["rows"][side], strict=True):
                    if n != q["clip"] or d["city"] != q["city"] or d["city"] != (cal if side == "calibration" else held):
                        raise ValueError("Fixed outcome-free forecast alignment changed")
                    row_scores = dict(q["score"], **scores(nets, d, path, ix, f["identity"]["cut"][0]))
                    rows[side].append(dict(clip=str(n), city=d["city"], score=row_scores))
            base.immutable(p, dict(identity=ident, rows=rows, training=training, fits={a: r for a, (_, r) in nets.items()}))
        z = study.read(p)
        if z["identity"] != ident:
            raise ValueError("Wrong rank fold")
        for r in z["fits"].values():
            study.checked(r["checkpoint"])
        done[name] = study.artifact(p)
        beat("freeze", completed=len(done), total=len(parent["folds"]))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, calibration_labels_used=False,
                                          held_labels_used=False, reserved_outcomes_opened=False))


def score(key, state, previous):
    parent = state[8]
    p = OUT/"freeze.json"
    f = study.read(p)
    if (f["registration"] != key or any(f[k] for k in ("calibration_labels_used", "held_labels_used", "reserved_outcomes_opened"))
            or set(f["folds"]) != set(parent["folds"])):
        raise ValueError("Require frozen score heads")
    frozen = {n: study.read(study.checked(ref)) for n, ref in f["folds"].items()}
    rules = {}
    for n, z in frozen.items():
        rows = list(support.prior.role_rows(z, parent["folds"][n], n, "calibration"))
        rules[n] = {a: gate.calibrate(rows, a) for a in ARMS}
    cp = OUT/"calibration.json"
    base.immutable(cp, dict(freeze=study.artifact(p), rules=rules, held_receipts_used=False))
    held, matched, train = [], [], Counter()
    for n, z in frozen.items():
        pf = parent["folds"][n]
        ann = study.read(support.LOCAL/f"{n}.support.json")
        if ann["parent"] != pf or z["identity"]["source"] != pf:
            raise ValueError("Wrong outcome receipt")
        train.update(z["training"])
        rows = list(support.prior.role_rows(z, pf, n, "held"))
        for r, b in zip(rows, ann["cells"], strict=True):
            if r["clip"] != b["clip"] or r["city"] != z["identity"]["held"]:
                raise ValueError("Wrong held alignment")
            r.update(support=b["support"], harm=b["cause"] == "observed_positive_harm", cause=b["cause"])
            r["release"] = {a: rules[n][a]["threshold"] is not None and r["scores"][a] >= rules[n][a]["threshold"] for a in ARMS}
        x = gate.matched_rows(rows, ARMS, z["identity"]["held"])
        if x:
            matched.append(x)
        held.extend(rows)
    summary = {a: support.prior.held_metrics(held, a) for a in ARMS}
    if len(held) != previous["views"] or any(summary[a] != previous["held"][a] for a in prior.ARMS):
        raise ValueError("Frozen comparator did not replay")
    counts = {a: {k: sum(r["arms"][a][k] for r in matched) for k in gate.FIELDS} for a in ARMS}
    if any(counts[a][k] != previous["matched_top5"]["counts"][a][k]
           for a in prior.ARMS for k in previous["matched_top5"]["counts"][a]):
        raise ValueError("Frozen same-count comparison changed")
    contrasts = {f"{a}_minus_{b}": {k: av.interval(matched,
        lambda r: 100*(r["arms"][a][k]-r["arms"][b][k])/r["n"]) for k in gate.FIELDS}
        for a, b in (("strict_point", "actor_support"), ("strict_rank", "actor_support"), ("strict_rank", "strict_point"))}
    out = dict(views=len(held), scenarios=len({r["clip"] for r in held}), train=train, held=summary,
        matched=dict(n=sum(r["n"] for r in matched), counts=counts, paired_pp=contrasts),
        calibration_pass={a: sum(r[a]["accepted"] for r in rules.values()) for a in ARMS},
        release_causes={a: Counter(r["cause"] for r in held if r["release"][a]) for a in ARMS},
        freeze=study.artifact(p), calibration=study.artifact(cp), result_source="fresh_run",
        outcome_source="cached_verified", independent_confirmation=False, reserved_outcomes_opened=False,
        original_2pct=True, deployment_changed=False)
    base.immutable(OUT/"summary.json", out)
    print(study.json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("fit", "score"))
    phase = ap.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, state, hf, previous = register()
        if phase == "fit":
            freeze(key, state, hf)
        else:
            score(key, state, previous)


if __name__ == "__main__":
    main()
