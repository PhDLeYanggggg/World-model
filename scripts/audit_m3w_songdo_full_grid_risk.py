"""Evaluate a complete-grid all-risk sensitivity on the original scored flights."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_songdo_recipient_attribution as prior
from src.evaluation.m3w_full_grid_risk import assess

primary, np = prior.primary, prior.np
OUT = prior.OUT.parent/'songdo_full_grid_all_risk_v1'
ARMS = (*prior.CONTROLS, 'joint')


def read_flight(data, name, archived):
    meta, cache, prefix, bank = data.observed(name)
    p = primary.checked(primary.PACKETS/f'{Path(name).stem}.npz', data.proposals[name]['packet_sha256'])
    with (np.load(prefix, allow_pickle=False) as x, np.load(bank, allow_pickle=False) as b,
          np.load(p, allow_pickle=False) as z):
        y, v = data.targets(name, cache, x, meta)
        cv = b['cv']
        np.testing.assert_array_equal(x['offsets'], b['offsets'])
        result = []
        for seed in primary.protocol.SEEDS:
            old = archived['scores'][str(seed)]
            for a in ARMS:
                key = f'seed{seed}_{a}'
                rows, path = z[key+'_rows'], z[key+'_paths']
                r = assess(cv[rows], path, y[rows], v[rows])
                risk = old['matched']['arms'][a]['risk']
                archived_ratio, actual = risk['completion_all_risk_upper'], r['original_all']
                if (archived_ratio is None) != (actual is None):
                    raise ValueError('Original all-risk denominator changed')
                if actual is not None:
                    np.testing.assert_allclose(actual, archived_ratio, atol=1e-12, rtol=0.)
                full = r['full_grid_upper']
                if not (r['partial_tracks'] or r['unknown_tracks']) and actual is not None:
                    np.testing.assert_allclose(actual, full, atol=1e-12, rtol=0.)
                result.append(dict(arm=a, edited=old['released'][a]['edited'], original_supported=risk['supported'],
                    original_all_pass=actual is not None and actual <= .02, full_grid_defined=full is not None,
                    full_grid_pass=full is not None and full <= .02,
                    partial_or_unknown=bool(r['partial_tracks'] or r['unknown_tracks'])))
    return result


def counts(rows):
    c = Counter(total=len(rows))
    for r in rows:
        c.update(original_all_pass=int(r['original_all_pass']), original_three_checks=int(r['original_supported']),
                 full_grid_all_pass=int(r['full_grid_pass']), full_grid_undefined=int(not r['full_grid_defined']),
                 partial_or_unknown=int(r['partial_or_unknown']),
                 original_three_and_full_grid_all=int(r['original_supported'] and r['full_grid_pass']),
                 original_three_with_adverse_completion=int(r['original_supported'] and r['full_grid_defined'] and not r['full_grid_pass']),
                 original_three_with_partial_or_unknown=int(r['original_supported'] and r['partial_or_unknown']))
    return dict(c)


def run():
    data, groups, hashes, bind, original = prior.check_scope()
    results = []
    for i, name in enumerate(sorted(groups), 1):
        old = prior.verified_flight(prior.published.RUN/'confirmation'/f'{Path(name).stem}.json', hashes[name])
        results.extend(read_flight(data, name, old))
        if i % 20 == 0:
            print(f'full-grid risk: {i}/160', flush=True)
    table = {}
    for arm in ARMS:
        rows = [r for r in results if r['arm'] == arm]
        matched, released = counts(rows), counts([r for r in rows if r['edited']])
        if (matched['total'] != 480 or matched['original_three_checks'] != original['matched'][arm]['risk']['original_risk_pass']
                or released['total'] != original['released'][arm]['risk']['edited']
                or released.get('original_three_checks', 0) != original['released'][arm]['risk']['original_risk_pass']):
            raise ValueError('Original matched or released counts changed')
        table[arm] = dict(matched=matched, actual_releases=released, abstained=480-released['total'])
    row = dict(result_source='fresh_retrospective_full_grid_all_risk_sensitivity', binding=bind,
               results=table, flights=len(groups), components=len(set(groups.values())), seeds=primary.protocol.SEEDS,
               original_2pct_unchanged=True, original_scored_outcomes_reused=True, new_external_cohort_opened=False,
               calibration_outcomes_read=False, deployment_changed=False, fit=False)
    prior.storage.write_json(OUT/'readout.json', row)
    print(table, flush=True)


if __name__ == '__main__':
    run()
