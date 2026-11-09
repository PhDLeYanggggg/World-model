"""Directly pair frozen anchor-ranker predictions under the existing scorers."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_vehicle_anchor_rank as rank

np, old, pair, base = rank.np, rank.old, rank.pair, rank.base
CONTRASTS = ('dual_only', 'joint')


def intervals(rows, sites, draws):
    values = np.array([np.mean([v for site, v in rows if site == s]) for s in sites])
    return dict(site_means=values.tolist(), mean=float(values.mean()),
                ci95=np.quantile(values[draws].mean(1), [.025, .975]).tolist())


def run():
    key, source, parent, _, _ = rank.register()
    fpath = rank.OUT/'freeze.json'
    f, scored = old.read(fpath), old.read(rank.OUT/'readout.json')
    if (f['registration'] != key or scored['registration'] != key or scored['freeze'] != rank.audit.sha(fpath)
            or set(f['cells']) != set(parent['cells']) or not f['held_choices_frozen']):
        raise ValueError('Expected the same completed frozen choices and score')
    data = base.parent.load(source)
    paired = {a: [] for a in CONTRASTS}
    active = {a: dict(directions=0, all_full_win=0, admitted=0, risk_pass=0, full_win=0, missing=0)
              for a in rank.FIT}
    seen = 0
    for name, c in f['cells'].items():
        for s, r in c['packets'].items():
            rule, d = c['rules'][s], data[s]
            if s == rule['calibration_site'] or s in parent['cells'][name]['fit_sites']:
                raise ValueError('Overlap in evaluation roles')
            with rank.audit.prior.packet(rank.LOCAL, r) as z:
                mass = float(d['weight'] @ np.where(d['valid'], np.linalg.norm(
                    z['cv']-d['future'], axis=-1), 0.).sum(1))
                p = {a: pair.prediction(z, a) for a in rank.ARMS}
                for a in CONTRASTS:
                    paired[a].append((s, pair.paired(p['relative'], p[a], d, z['bounds'], mass)))
                for a in rank.FIT:
                    threshold = rule['arms'][a]['threshold']
                    if threshold is None:
                        continue
                    use = z[f'{a}_scores'] >= threshold
                    if not use.any():
                        continue
                    status, full, _, _ = rank.compare(p[a], p['additive'], z, d, c['cut'])
                    counts = active[a]
                    counts['directions'] += 1
                    counts['all_full_win'] += int(full[use].all())
                    counts['admitted'] += int(use.sum())
                    counts['risk_pass'] += int(status['risk'][use].sum())
                    counts['full_win'] += int(full[use].sum())
                    counts['missing'] += int(status['missing'][use].sum())
            seen += 1
    if seen != scored['directions'] or any(
            active[a]['admitted'] != scored['totals'][a]['admitted']
            or active[a]['full_win'] != scored['totals'][a]['admitted_full_win'] for a in active):
        raise ValueError('Release or direction count changed')
    sites = base.SITES
    draws = np.random.default_rng(2917).integers(len(sites), size=(3000, len(sites)))
    results = {a: {k: intervals([(s, v[k]) for s, v in rows], sites, draws) for k in rows[0][1]}
               for a, rows in paired.items()}
    old.write(rank.OUT/'paired_readout.json', dict(freeze=scored['freeze'], directions=seen,
        result_source='fresh_run', original_2pct_unchanged=True, external_outcomes_read=False,
        independent_confirmation=False, active=active, relative_minus=results))
    print(active, results, flush=True)


if __name__ == '__main__':
    rank.torch.set_num_threads(4)
    rank.torch.set_num_interop_threads(1)
    run()
