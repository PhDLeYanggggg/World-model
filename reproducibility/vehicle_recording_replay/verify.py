"""Replay the frozen two-edit vehicle comparison from recording scalars."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import fsum, isfinite
from pathlib import Path
from statistics import mean


ARMS = ('additive', 'joint', 'dual_only', 'relative')
FITTED = ARMS[2:]
DIRECT = ('dual_only', 'joint')
SEEDS = (17, 29, 43)


def delta(terms):
    known, missing = terms
    if not all(isfinite(x) for x in terms) or missing < 0:
        raise ValueError('Invalid paired error terms')
    return known-missing


def relative(terms):
    known, missing, count = terms
    if not isinstance(count, int) or count <= 0:
        raise ValueError('Invalid full pair-step denominator')
    return delta((known, missing))/count


def qualification(parts):
    all_ref, easy_ref, full_easy, harm, easy_harm, debit, signed = parts['risk']
    if not all(isfinite(x) for x in parts['risk']) or min(parts['risk'][:-1]) < 0:
        raise ValueError('Invalid recording risk components')
    risk = (all_ref > 0 and easy_ref > 0 and full_easy > 0 and
            (harm+debit)/all_ref <= .02 and (easy_harm+debit)/easy_ref <= .02 and signed/full_easy <= .02)
    dual = risk and delta(parts['cv']) > 0 and delta(parts['ols']) > 0
    t, r = delta(parts['add']), relative(parts['pair'])
    return risk, dual, dual and t >= -1e-10 and r > 1e-10, easy_ref <= 0, t, r


def prefix(scored):
    """Largest tie-complete score prefix in which every calibration row qualifies."""
    eligible = [(s, v) for s, v in scored if s is not None]
    if not eligible:
        return None, 0
    failing = [s for s, v in eligible if not v]
    bound = max(failing, default=float('-inf'))
    kept = [s for s, _ in eligible if s > bound]
    return (min(kept), len(kept)) if kept else (None, 0)


def top_mass(scored, k):
    by_score = defaultdict(list)
    for score, qualified in scored:
        if score is not None:
            by_score[score].append(qualified)
    n, hit = 0, 0.
    for _, group in sorted(by_score.items(), reverse=True):
        take = min(k-n, len(group))
        n += take
        hit += take*mean(group)
        if n == k:
            break
    return hit, n


def interval(rows, draws):
    sites = [mean(rows[i]) for i in range(7)]
    sims = sorted(mean(sites[i] for i in b) for b in draws)

    def quantile(q):
        x = q*(len(sims)-1)
        k = int(x)
        return sims[k]+(x-k)*(sims[min(k+1, len(sims)-1)]-sims[k])

    return dict(site_means=sites, mean=mean(sites), ci95=[quantile(.025), quantile(.975)])


def replay(data):
    cells = {(c['site'], c['calibration'], c['seed']): c for c in data['directions']}
    roster = {(s, c, seed) for s in range(7) for c in range(7) if s != c for seed in SEEDS}
    draws = data['source_resamples']
    if (set(cells) != roster or len(data['directions']) != len(roster) or len(draws) != 3000
            or any(len(d) != 7 or any(type(i) is not int or not 0 <= i < 7 for i in d) for d in draws)):
        raise ValueError('Incomplete frozen roster or invalid source resamples')
    per_record = defaultdict(set)
    totals = {a: Counter() for a in ARMS}
    active = {a: Counter() for a in FITTED}
    triage = {a: Counter() for a in FITTED}
    by_seed = {a: {seed: Counter() for seed in SEEDS} for a in ARMS}
    group = {a: defaultdict(lambda: defaultdict(list)) for a in ARMS}
    paired = {a: defaultdict(lambda: defaultdict(list)) for a in DIRECT}
    for (site, cal, seed), cell in cells.items():
        rows = cell['recordings']
        if not rows or cell['cv_mass'] <= 0 or not isfinite(cell['cv_mass']):
            raise ValueError('Undefined common normalization mass')
        ids = [r['id'] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate recording in one evaluation direction')
        for i in ids:
            per_record[i].add((site, cal, seed))
        val = cells[cal, site, seed]['recordings']
        n = len(rows)
        scale = 100./cell['cv_mass']
        for a in ARMS:
            q = [qualification(r['arms'][a]) for r in rows]
            risk, dual, win, missing, t, r = zip(*q, strict=True)
            count = dict(count=n, dual=sum(dual), full_win=sum(win))
            g = dict(ade_lower_pp=fsum(t)*scale, relative_lower_native=mean(r))
            if a in FITTED:
                scores = [row['arms'][a]['score'] for row in rows]
                if any(s is not None and not isfinite(s) for s in scores):
                    raise ValueError('Nonfinite rank score')
                tcal, ncal = prefix((x['arms'][a]['score'], qualification(x['arms'][a])[2]) for x in val)
                if cell['thresholds'][a] != [tcal, ncal]:
                    raise ValueError('Frozen threshold does not replay on calibration side')
                take = [i for i, s in enumerate(scores) if s is not None and tcal is not None and s >= tcal]
                count.update(eligible=sum(s is not None for s in scores), admitted=len(take),
                             admitted_dual=sum(dual[i] for i in take), admitted_full_win=sum(win[i] for i in take),
                             admitted_missing=sum(missing[i] for i in take))
                g.update(gated_ade_lower_pp=fsum(t[i] for i in take)*scale,
                         gated_relative_lower_native=fsum(r[i] for i in take)/n)
                active[a].update(directions=bool(take), all_full_win=bool(take) and all(win[i] for i in take),
                                 admitted=len(take), risk_pass=sum(risk[i] for i in take),
                                 full_win=sum(win[i] for i in take), missing=sum(missing[i] for i in take))
                top_hit, top_n = top_mass(zip(scores, win, strict=True), 5)
                triage[a].update(calibration_prefix_directions=tcal is not None,
                                 top5_full_win_mass=top_hit, top5_candidates=top_n,
                                 matched_full_win=sum(win), matched_candidates=n)
            totals[a].update(count)
            by_seed[a][seed].update(count)
            for k, v in g.items():
                group[a][k][site].append(v)
        for a in DIRECT:
            terms = [row['direct'][a] for row in rows]
            paired[a]['ade_available_pp'][site].append(scale*fsum(x[0][0] for x in terms))
            paired[a]['ade_lower_pp'][site].append(scale*fsum(delta(x[0]) for x in terms))
            paired[a]['relative_lower_native'][site].append(mean(relative(x[1]) for x in terms))
    if (len(per_record) != 493 or set(per_record) != set(range(493))
            or any(len(v) != 18 or len({s for s, _, _ in v}) != 1 for v in per_record.values())):
        raise ValueError('Recordings do not repeat exactly over three seeds and six calibration choices')
    return dict(totals=totals, active=active, by_seed=by_seed, ranking_diagnostic=triage,
                intervals={a: {k: interval(rows, draws) for k, rows in metrics.items()} for a, metrics in group.items()},
                relative_minus={a: {k: interval(rows, draws) for k, rows in metrics.items()} for a, metrics in paired.items()},
                locations=7, unique_recordings=len(per_record), evaluation_directions=len(cells),
                recording_views=sum(map(len, per_record.values())))


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/'manifest.json').read_text())
    if set(manifest) != {'README.md', 'evidence.json', 'verify.py'}:
        raise ValueError('Unexpected archive members')
    for name, expected in manifest.items():
        if sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Archive member hash differs: '+name)
    print(json.dumps(replay(json.loads((root/'evidence.json').read_text())), indent=2))


if __name__ == '__main__':
    main()
