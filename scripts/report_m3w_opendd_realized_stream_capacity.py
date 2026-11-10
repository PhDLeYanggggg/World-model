"""Decompose the frozen OpenDD budget comparison and resample held locations."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_opendd_realized_stream_capacity as study


def run():
    run, np, broad = study.run, study.np, study.broad
    p = study.OUT/'readout.json'
    score = run.read(p)
    reg = run.sha(study.OUT/'registration.json')
    old = run.sha(broad.OUT/'registration.json')
    if score['registration'] != reg or not score['original_2pct'] or score['confirmation_access']:
        raise ValueError('Expected frozen development-only exact-count results')
    changes = Counter()
    for site, rows in score['by_site'].items():
        for v in rows:
            seed = v['seed']
            for i in range(v['views']):
                r = run.read(study.PRIVATE/f'{site}_{seed}_{i:04d}.json')
                baseline = run.read(broad.PRIVATE/f'{site}_{i:04d}.json')
                if (r['identity'] != dict(registration=reg, site=site, ordinal=i, seed=seed)
                        or (baseline['registration'], baseline['site'], baseline['ordinal']) != (old, site, i)):
                    raise ValueError('Cannot match original and realized-budget views')
                earlier = next(x['arms']['nested_stream']['status'] for x in baseline['entries'] if x['seed'] == seed)
                if earlier == study.PASS and r['result']['status'] != study.PASS:
                    changes['quota_witness_lost_at_zero_actual_edits' if r['total'] == 0
                            else 'quota_witness_lost_at_positive_exact_count'] += 1
                if r['result']['status'] == study.PASS and not r['actual']:
                    changes['exact_count_witness_missed_by_online_timing'] += 1
    t = score['summary']
    if (sum(changes[k] for k in ('quota_witness_lost_at_zero_actual_edits', 'quota_witness_lost_at_positive_exact_count'))
            != t['broad_pass']-t['exact_pass'] or changes['exact_count_witness_missed_by_online_timing']
            != t['exact_pass']-t['online_pass']):
        raise ValueError('Timing and count changes do not partition the archived counts')
    draws = np.random.default_rng(817).integers(len(score['by_site']), size=(3000, len(score['by_site'])))

    def paired(a, b):
        v = np.array([np.mean([100.*(int(row[a])-int(row[b]))/row['views'] for row in rows])
                      for rows in score['by_site'].values()])
        return dict(point_pp=float(v.mean()), ci95_pp=np.quantile(v[draws].mean(1), [.025, .975]).tolist(),
                    locations_positive=int((v > 0.).sum()), locations=len(v))

    return dict(summary=t, changes=dict(changes), exact_minus_online=paired('exact_pass', 'online_pass'),
                quota_minus_exact=paired('broad_pass', 'exact_pass'), bootstrap_draws=len(draws),
                readout_sha256=run.sha(p), original_2pct=True, confirmation_access=False,
                hindsight_only=True, no_fitting=True, submission_ready=False)


if __name__ == '__main__':
    result = run()
    study.run.once(study.OUT/'decomposition.json', study.run.stable(result))
    print(result)
