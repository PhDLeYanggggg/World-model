"""Retrospective attribution with identical recording totals and token rules."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_recording_count as study

run, np, torch, parent = study.run, study.np, study.torch, study.parent


def exact_score_plan(scores, legal, total):
    """Maximize summed logit increments at the required total, without labels."""
    if total == 0:
        return np.zeros(len(scores), dtype=int)
    if study.RATE != .2:
        raise ValueError("The integer token encoding is defined only for rate 0.2")
    val = np.full((total+1, 11), -np.inf)
    val[0, 0] = 0.
    history = np.full((len(scores), total+1, 11, 2), -1, dtype=np.int8)
    for t, (s, allowed) in enumerate(zip(scores, legal, strict=True)):
        nxt = np.full_like(val, -np.inf)
        for token in range(11):
            after = min(10, token+int(allowed[1]))
            for c in range(min(2, total, after//5)+1):
                if not allowed[c]:
                    continue
                n = total+1-c
                to = after-5*c
                cand = val[:n, token]+float(s[c]-s[0])
                chosen = cand > nxt[c:, to]
                nxt[c:, to][chosen] = cand[chosen]
                history[t, c:, to][chosen] = (token, c)
        val = nxt
    token = val[total].argmax()
    if not np.isfinite(val[total, token]):
        raise ValueError("The realized matched total must be reachable")
    count = np.zeros(len(scores), dtype=int)
    left = total
    for t in range(len(scores)-1, -1, -1):
        token, c = history[t, left, token]
        if c < 0:
            raise ValueError("Inconsistent token backtrace")
        count[t] = c
        left -= c
    if left or count.sum() != total:
        raise ValueError("Retrospective match changed the requested edit count")
    return count


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, a, _, _, ph = study.register()
    p = study.OUT / "analysis.json"
    h = study.OUT / "head_freeze.json"
    analysis, heads = run.read(p), run.read(h)
    if (analysis["registration"] != key or heads["registration"] != key or analysis["heads"] != run.sha(h)
            or analysis["confirmation_access"] or not analysis["original_2pct"]):
        raise ValueError("Require frozen recording count experiment")
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    result = {}
    for s, d in sites.items():
        file = parent.prior.prior.old.source.prior.PRIVATE / f"{s}_cohort.npz"
        with np.load(file, allow_pickle=False) as z:
            rec, starts = z["recording"], z["start"]
        bank = parent.prior.analytic(d)
        cv, ols, b = bank[:3]
        y, valid = d["y"], d["valid"]
        mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
        cuts = np.quantile(np.concatenate([v["cut_samples"] for k, v in sites.items() if k != s]), [.25, .75])
        rows = []
        for cell in a["by_site"][s]:
            seed = cell["seed"]
            rank = study.load(ph["heads"][f"{s}/{seed}/additive_rank"], lambda: study.prior.model.SwapValue(False), study.prior.STEPS)
            menu = study.menus(d, bank, rank)
            legal = menu["legal"]
            logits, online = {}, {}
            for k in ("local", "recording"):
                net = study.load(heads["count_heads"][f"count_{s}_{seed}_{k}"],
                                 lambda: study.model.CountHead(menu["x"].shape[-1]), study.HEAD_STEPS)
                with torch.inference_mode():
                    logits[k] = net(torch.from_numpy(menu["x"]), torch.from_numpy(legal)).numpy()
                online[k] = study.model.streaming_count(logits[k], legal, rec, starts, study.RATE)
            exact = {k: np.zeros(len(rec), dtype=int) for k in online}
            supported = 0
            for r in sorted(set(rec)):
                at = rec == r
                total = min(int(v[at].sum()) for v in online.values())
                if not total:
                    continue
                supported += 1
                for k in exact:
                    plan = exact_score_plan(logits[k][at], legal[at], total)
                    score = np.full_like(logits[k][at], -1.)
                    score[np.arange(len(plan)), plan] = 1.
                    np.testing.assert_array_equal(study.model.streaming_count(score, legal[at], rec[at], starts[at], study.RATE), plan)
                    exact[k][at] = plan
            paths, stats = {}, {}
            for k, count in exact.items():
                use = np.arange(2) < count[:, None]
                at = np.full(len(cv), -1, dtype=int)
                at[menu["row"][use]] = menu["arm"][use]
                paths[k] = parent.selected(cv, b, at)
                stats[k] = parent.score(d, cv, ols, paths[k], cuts)
            gain = float(100.*parent.transfer.paired(paths["local"], paths["recording"], y, valid)[1].sum()/mass)
            rows.append(dict(seed=seed, scored=stats, gain=gain, nonzero_matched_recordings=supported))
        result[s] = rows
    x = np.array([np.mean([r["gain"] for r in v]) for v in result.values()])
    draw = np.random.default_rng(817).integers(len(x), size=(3000, len(x)))
    cells = [r for v in result.values() for r in v]
    out = dict(gain=dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist()),
        matched_recordings=sum(r["nonzero_matched_recordings"] for r in cells),
        scored={k: {v: sum(r["scored"][k][v] for r in cells) for v in ("edits", "dual_positive", "risk_pass", "views", "undefined")}
                for k in ("local", "recording")})
    run.once(study.OUT / "matched_diagnostic.json", run.stable(dict(code=run.artifact(Path(__file__)), registration=key,
        analysis=run.artifact(p), by_site=result, summary=out, chronological_feasibility=True,
        sees_all_recording_histories_for_matching_only=True, deployment_inference=False, labels_not_used_for_matching=True,
        original_2pct=True, confirmation_access=False)))


if __name__ == "__main__":
    main()
