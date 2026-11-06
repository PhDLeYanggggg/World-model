"""Cross-check unchanged control estimands against the earlier complete audit."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_prefix_actor as run


def main():
    reg = run.read(run.REG); a = run.read(run.PUBLIC/'analysis.json')
    prior = run.read(run.checked(next(p for p in reg['parents'] if p['path'] == reg['config']['diagnosis'])))
    legacy = {(b['job'], s['source']): s for b in prior['banks'] for s in b['sources']}
    comparisons = 0
    for s in a['sources']:
        old = legacy[(s['job'], s['source'])]
        if old['rows'] != s['rows']:
            raise ValueError('Control roster changed')
        for row in s['summary']:
            if row['prefix'] not in (1, 3, 12):
                continue
            r = next(r for r in old['summary'] if r['prefix'] == row['prefix'])
            if (row['known_rows'] != r['known_rows'] or row['wholly_unknown_rows'] != r['wholly_unknown_rows']
                    or row['partial_rows'] != r['partial_prefix_rows']):
                raise ValueError('Control label support changed')
            run.np.testing.assert_allclose(row['control']['known_gain_percent'], r['known_gain_percent'], atol=1e-8, rtol=0)
            comparisons += 1
    if comparisons != 42 or sum(s['rows'] for s in a['sources']) != 125540:
        raise ValueError('All14sources and three shared prefixes required')
    value = dict(result_source='cached_verified', whole_roster_and_support_preserved=True,
        comparisons=42, all_control_query_balanced_gain_percent_equal_atol1e_8=True,
        model_selection_or_rule_change=False)
    path = run.PUBLIC/'control_readout_verification.json'
    if path.exists():
        run.exact(run.read(path), value)
    else:
        run.once(path, value)
    print('All42unchanged-control estimands match prior complete prefix audit.')


if __name__ == '__main__':
    main()
