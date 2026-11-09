"""Diagnose nested calibration and its transport to fixed OpenDD slots."""
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_nested_spline_choice as m
from src.evaluation.m3w_selection_forecast_readout import policy

control, run, np = m.control, m.run, m.np


def main():
    fp, rp = (m.OUT / n for n in ('freeze.json', 'readout.json'))
    f, r = map(run.read, (fp, rp))
    if (r['freeze'] != run.sha(fp) or r['confirmation_access']
            or r['independent_confirmation'] or not f['before_held_scoring']):
        raise ValueError('Require complete frozen development readout')
    reg, source, _, _, _, _, _ = m.register()
    if f['registration'] != reg:
        raise ValueError('Freeze registration mismatch')
    data = m.m.load(source)
    sites = set(data)
    lineage = {}
    for role in ('producer', 'selector'):
        lineage[role] = Counter()
        for item in f[f'{role}_fits'].values():
            i = item['identity']
            excluded, fit = set(i['excluded']), set(i['fit'])
            if (excluded & fit or excluded | fit != sites or i['registration'] != reg or i['arm'] != role):
                raise ValueError('Invalid nested checkpoint roles')
            lineage[role][str(len(excluded))] += 1
    diagnostics = {}
    for kind in m.KINDS:
        totals = Counter()
        for cal in f['calibration'].values():
            selected, table = cal[kind]
            good = [z for z in table if z['rule'] is not None and z['gain'] > 0.]
            totals['folds_with_rule'] += int(selected is not None)
            totals['folds_positive_before_risk'] += int(bool(good))
            totals['positive_grid_entries'] += len(good)
            totals['positive_admissible_entries'] += sum(z['admissible'] for z in good)
            for j, key in enumerate(('all', 'easy', 'signed')):
                totals[f'positive_entries_excluded_by_{key}'] += sum(z['risk'][j] > 1e-12 for z in good)
        diagnostics[kind] = dict(totals)
    geometry = m.m.donor_spline.prior.geometry
    group = m.m.donor_spline.prior.query_groups
    paired, roster, changed, tails = {}, {}, Counter(), {}
    for s, site in data.items():
        d = site['d']
        bank = control.parent.prior.analytic(d)
        cv = bank[0]
        proxy = m.proxy(d, m.packet(d, bank, cv))
        roster[s] = dict(proxy_slots=len(proxy), actual_slots=[])
        for c in (z for z in f['cells'] if z['held'] == s):
            with np.load(run.checked(c['paths']), allow_pickle=False) as z:
                paths = {k: z[k] for k in m.ARMS}
            a = np.stack([paths[k] for k in ('anchor', 'prefix_match', 'independent', 'joint', 'joint_same_rule')])
            for k, p in paths.items():
                metrics = policy(p, cv, cv, d['y'], d['valid'], d['rec'], easy_cut=c['cut'][0], hard_cut=c['cut'][1])
                row = tails.setdefault(k, dict(easy_signed_over_2=0, worst_easy_signed=None, hard_gain={}))
                signed = [r['known_easy_net_degradation'] for r in metrics['recordings'].values()
                          if r['known_easy_net_degradation'] is not None]
                row['easy_signed_over_2'] += sum(v > .02 for v in signed)
                if signed:
                    row['worst_easy_signed'] = max([*signed] + ([] if row['worst_easy_signed'] is None
                                                               else [row['worst_easy_signed']]))
                row['hard_gain'][s, c['seed']] = metrics['subsets']['hard']['gain_percent']
            roster[s]['actual_slots'].append(int((a[0] != cv).any((1,2)).sum()))
            changed['joint_vs_independent'] += int((a[3] != a[2]).any((1,2)).sum())
            changed['shared_vs_independent'] += int((a[4] != a[2]).any((1,2)).sum())
            known, lower, n, u = np.zeros(3), np.zeros(3), 0, 0
            for g in group(d['query']):
                _, x, _, (steps, obs, _, _) = geometry.query(d['h'][g], a[:, g], d['y'][g],
                    d['valid'][g], ((0, 3), (1, 3), (2, 4)))
                if obs:
                    known += x[:, 0]*obs
                    n += obs
                if steps:
                    lower += x[:, 1]*steps
                    u += steps
            paired[s, c['seed']] = np.stack((known/max(n,1) if n else np.full(3,np.nan),
                                             lower/max(u,1) if u else np.full(3,np.nan)), axis=1)
    g = sorted(sites)
    rng = np.random.default_rng(817)
    draw = rng.integers(len(g), size=(3000,len(g)))
    x = np.array([[paired[s,k] for k in m.m.SEEDS] for s in g])
    pairs = {k: {n:geometry.bootstrap(x[:,:,j,i],draw) for i,n in enumerate(('observed','lower'))}
        for j,k in enumerate(('joint_vs_anchor','joint_vs_prefix','grouped_at_same_rule'))}
    for row in tails.values():
        hard = row.pop('hard_gain')
        row['observed_hard_gain_vs_cv_percent'] = geometry.bootstrap([[hard[s,k] for k in m.m.SEEDS] for s in g], draw)
        if row['worst_easy_signed'] is not None:
            row['worst_easy_signed_percent'] = 100.*row.pop('worst_easy_signed')
    out = dict(parent=run.artifact(rp), code=run.artifact(Path(__file__)),
        checkpoint_exclusions={k:dict(v) for k,v in lineage.items()}, calibration=diagnostics,
        roster=roster, changed=dict(changed), pair_displacement_publisher_coordinates=pairs, risk_tail=tails,
        diagnostic_only=True, confirmation_access=False, independent_confirmation=False)
    run.once(m.OUT/'diagnosis.json', run.stable(out))
    print(dict(calibration=diagnostics, changed=dict(changed)), flush=True)


if __name__ == '__main__':
    main()
