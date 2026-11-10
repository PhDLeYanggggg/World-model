"""Bound risk and both accuracy contrasts on the already scored WEPDTOF roster."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_wepdtof_frozen_capacity as frozen
from scripts import run_m3w_wepdtof_external_readout as prior
from src.evaluation.m3w_full_grid_risk import assess, assess_easy

np, study, base = prior.np, prior.study, prior.base
OUT = frozen.PUBLIC/'wepdtof_full_grid_risk_v1'
ARMS = frozen.MATCHED
EPS = 1e-12


def gain_bounds(ref, proposal, target, valid, weights, mass):
    gain, low = study.paired(ref, proposal, target, valid)
    if mass <= 0.:
        return (None, None, None)
    return tuple(float(100*(weights @ v.sum(1))/mass) for v in (gain, low, 2*gain-low))


def replay(value, previous):
    if (value is None) != (previous is None):
        raise ValueError('Original support changed')
    if value is not None:
        np.testing.assert_allclose(value, previous, atol=EPS, rtol=EPS)


def inspect(cv, ols, proposal, target, valid, weights, mass, archived, easy_cut):
    edited = (proposal != cv).any(axis=(1, 2))
    n = int(edited.sum())
    old = archived.get('risk', {})
    if n != old.get('selected', 0):
        raise ValueError('Original edit count changed')
    if not n:
        if old.get('supported') or archived.get('dual_pass'):
            raise ValueError('Abstention was counted as a pass')
        return dict(edits=0)
    args = tuple(x[edited] for x in (cv, proposal, target, valid))
    all_risk = assess(*args)
    easy = assess_easy(*args, easy_cut=easy_cut)
    replay(all_risk['original_all'], old['completion_all_risk_upper'])
    replay(easy['original_easy'], old['completion_easy_risk_upper'])
    signed = old['known_easy_net_degradation']
    if easy['original_easy'] is not None and (signed is None or signed > easy['original_easy']+EPS):
        raise ValueError('Original signed easy harm exceeds its positive-harm bound')
    bounds = []
    for label, ref in (('cv', cv), ('ols4', ols)):
        observed, lo, hi = gain_bounds(ref, proposal, target, valid, weights, mass)
        replay(observed, archived[f'known_vs_{label}'])
        replay(lo, archived[f'lower_vs_{label}'])
        bounds.append((lo, hi))
    full_all = all_risk['full_grid_upper']
    full_risk_pass = full_all is not None and full_all <= .02 and easy['selected_easy_pass_for_all']
    # Use a small numerical dead zone for strict utility conclusions.
    utility_out = any(hi is not None and hi < -EPS for _, hi in bounds)
    utility_guaranteed = all(lo is not None and lo > EPS for lo, _ in bounds)
    return dict(edits=n, original_easy_support=easy['original_easy'] is not None,
                original_risk_pass=old['supported'], original_dual_pass=archived['dual_pass'],
                full_easy_support=easy['easy_support_guaranteed'], full_risk_pass=full_risk_pass,
                original_and_full_risk_pass=old['supported'] and full_risk_pass,
                original_pass_utility_excluded=old['supported'] and utility_out,
                full_pass_utility_excluded=full_risk_pass and utility_out,
                utility_excluded=utility_out, utility_guaranteed=utility_guaranteed,
                full_grid_dual_guaranteed=full_risk_pass and utility_guaranteed)


def paired_rate(groups, arm, field):
    values = []
    for group in groups:
        edited = [row for row in group if row['fixed_joint']['edits']]
        if edited:
            values.append(np.mean([int(row['fixed_joint'][field])-int(row[arm][field]) for row in edited]))
    a = np.asarray(values)
    if len(a) < 2:
        raise ValueError('Too few matched original sources')
    draws = np.random.default_rng(82317).choice(a, size=(3000, len(a))).mean(1)
    return dict(point_pp=float(100*a.mean()), ci95_pp=(100*np.quantile(draws, [.025, .975])).tolist(),
                estimable_sources=len(a), bootstrap_draws=len(draws))


def run():
    scores = json.loads(frozen.checked(study.OUT/'external_readout.json', frozen.READOUT))
    freeze = json.loads(frozen.checked(study.OUT/'augmented_freeze.json', frozen.FREEZE))
    if (scores['forecast_freeze'] != frozen.FREEZE or not scores['external_scoring_completed']
            or scores['external_tuning'] or scores['dronecrowd_confirmation_access'] or not scores['original_2pct']
            or not freeze['outcomes_unopened'] or not freeze['original_2pct']):
        raise ValueError('Require frozen predictions and the completed original readout')
    reg = study.OUT/'evaluation_registration.json'
    frozen.checked(reg, freeze['evaluation_registration'])
    cut_record = study.read(reg)['pilot']
    cut_path = study.checked(cut_record)
    easy_cut = study.read(cut_path)['cut_px'][0]
    camera, windows = study.frozen_camera()
    records = {r['clip']: r for r in scores['rows']}
    if (len(records) != 15 or len(freeze['rows']) != 15 or set(records) != set(windows)
            or set(records) != {r['clip'] for r in freeze['rows']}
            or len({r['source'] for r in records.values()}) != 13 or study.checked(freeze['camera']) != camera):
        raise ValueError('Original external roster changed')
    by_source = defaultdict(list)
    for i, f in enumerate(freeze['rows'], 1):
        clip, source = f['clip'], f['source']
        old = records[clip]
        h, q, ids = study.camera_arrays(windows[clip])
        if old['source'] != source or old['online'] != len(q) or old['freeze'] != frozen.FREEZE:
            raise ValueError('Original recording receipt disagrees with forecast packet')
        with np.load(study.checked(f['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['q'], q)
            np.testing.assert_array_equal(z['ids'], ids)
            y, valid, matched, histories = study.outcomes(windows[clip], h, q, ids, study.references(clip))
            if (int(matched.sum()), int(valid.all(1).sum()), int((~valid.any(1)).sum()), histories) != tuple(
                    old[k] for k in ('matched', 'full', 'missing', 'reference_histories')):
                raise ValueError('Original label support changed')
            _, inv, counts = np.unique(q, return_inverse=True, return_counts=True)
            w = 1./counts[inv]
            for seed in frozen.SEEDS:
                cv, ols = (z[f'{seed}_{k}'] for k in ('cv', 'ols4'))
                mass = float(w @ np.where(valid, np.linalg.norm(cv-y, axis=-1), 0.).sum(1))
                old_seed = old['seeds'][str(seed)]
                replay(mass, old_seed['reference_mass'])
                row = {arm: inspect(cv, ols, z[f'{seed}_{arm}'], y, valid, w, mass,
                                    old_seed['arms'].get(arm, {}), easy_cut) for arm in ARMS}
                if len({r['edits'] for r in row.values()}) != 1:
                    raise ValueError('Frozen matched edit counts differ')
                by_source[source].append(row)
        print(f'WEPDTOF sensitivity: {i}/15 clips', flush=True)
    groups = [by_source[k] for k in sorted(by_source)]
    result = {}
    for arm in ARMS:
        rows = [r[arm] for g in groups for r in g]
        c = Counter(views=len(rows), edited_views=sum(r['edits'] > 0 for r in rows))
        for r in rows:
            c.update(r)
        summary = scores['summary']['arms'][arm]
        if (c['views'], c['edits'], c['original_risk_pass'], c['original_dual_pass']) != (
                45, summary['edits'], summary['risk_passes'], summary['dual_passes']):
            raise ValueError('Original aggregate did not replay')
        result[arm] = dict(c)
    return dict(result_source='fresh_retrospective_full_grid_risk_and_utility_sensitivity',
                bindings=dict(archived_readout_sha256=frozen.READOUT, forecast_freeze_sha256=frozen.FREEZE,
                              registration_sha256=freeze['evaluation_registration'], pilot_sha256=cut_record['sha256'],
                              protocol_sha256=base.sha(OUT/'protocol.md'), driver_sha256=base.sha(Path(__file__))),
                clips=len(records), sources=len(groups), seeds=frozen.SEEDS, easy_cut_px=easy_cut, limit=.02,
                arms=result, joint_minus_independent={field: paired_rate(groups, 'fixed_independent', field)
                    for field in ('full_easy_support', 'full_risk_pass', 'utility_excluded')},
                original_rule_unchanged=True, original_scored_outcomes_reused=True, new_confirmation=False,
                external_tuning=False, fit=False, deployment_changed=False, submission_ready=False)


if __name__ == '__main__':
    out = run()
    base.immutable(OUT/'readout.json', out)
    print(json.dumps(out['arms'], indent=2), flush=True)
