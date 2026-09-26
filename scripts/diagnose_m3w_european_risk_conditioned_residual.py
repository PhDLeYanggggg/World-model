"""Descriptive post-freeze error identities, never a held-tuned correction."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_european_risk_conditioned_residual as run
from scripts.diagnose_m3w_european_nested_residual import accounting


def main():
    cfg, identity = run.registration(); run.check_freeze()
    assert json.loads((run.PUBLIC/'completion_checks.json').read_text())['all_passed']
    rows = []; by_cache = {}
    for v in run.nested.views(identity['parent']['parent']):
        name = v['g']['group']+'_'+v['pair']
        if name not in by_cache: by_cache = {name: run.base.pair_inputs(v['g'], v['data'], v['pairs'], v['pair'])[2]}
        cv = v['data']['baseline_ade'][v['held_ids'], 1]
        y = run.source.tail.diagnostic.event_targets(by_cache[name][v['te']], cv, v['pr']['positive_easy_cut'])
        f = next(f for f in json.loads((run.PUBLIC/'groups'/(name+'.json')).read_text())['folds'] if f['held'] == v['outer'])
        assert run.array_hash(y) == f['target_sha256']
        with np.load(run.context_run.frozen_directory(v['tag'], 'original')/'scores.npz', allow_pickle=False) as z:
            original = z['scores'].copy()
        with np.load(run.parent.PRIVATE/'probes'/v['tag']/'scores.npz', allow_pickle=False) as z:
            common = z['oof__context_bias'].copy()
        env = v['pairs']['B'][v['pair']][1][v['te']]
        with np.load(run.PRIVATE/'probes'/v['tag']/'scores.npz', allow_pickle=False) as z:
            for variant in cfg['variants']:
                for arm in cfg['arms']:
                    key = variant+'__'+arm; q = z[key]
                    for control, p in (('original', original), ('common_context', common)):
                        a = accounting(p[:, 3], q[:, 3], y[:, 3], env > 0)
                        np.testing.assert_allclose(a['corrected_MSE'], f['metrics']['risk__'+key]['envelope_positive']['harm_MSE'], rtol=1e-12, atol=1e-12)
                        rows.append(dict(tag=v['tag'], pair=v['pair'], variant=variant, arm=arm, control=control, **a))
    assert len(rows) == 1728
    private = run.PRIVATE/'error_accounting.json'
    run.immutable_json(private, dict(rows=rows, predictions_changed=False, held_guided_selection=False))
    kinds = ['improved', 'unchanged', 'wrong_aggregate_direction', 'useful_direction_excess_magnitude']
    summaries = []
    for pair in cfg['pairs']:
        for variant in cfg['variants']:
            for arm in cfg['arms']:
                for control in ('original', 'common_context'):
                    rr = [r for r in rows if (r['pair'], r['variant'], r['arm'], r['control']) == (pair, variant, arm, control)]
                    assert len(rr) == 72
                    summaries.append(dict(pair=pair, variant=variant, arm=arm, control=control,
                        counts={k: sum(r['description'] == k for r in rr) for k in kinds}))
    doc = dict(result_source='fresh_run_post_freeze_descriptive_error_accounting', exact_checks=1728,
        summaries=summaries, details=run.artifact(private), independent_views=False,
        predictions_changed=False, held_guided_model_selection=False, causal_root_cause_proven=False)
    run.immutable_json(run.PUBLIC/'error_accounting.json', doc)
    lines = ['# Descriptive Error Accounting', '',
        'For old error e and applied shift d, MSE change = 2 mean(e*d) + mean(d^2).',
        'All 1,728 identities agree with direct arithmetic. No shrinkage or other parameter is fitted from these outcomes.',
        'This explains the implemented error difference algebraically, not its causal origin.', '',
        '| Inputs / bank / arm / control | Improved | Unchanged | Non-helpful aggregate direction | Helpful direction, excess size |',
        '|---|---:|---:|---:|---:|']
    for s in summaries:
        lines.append('| '+' / '.join(s[k] for k in ('pair', 'variant', 'arm', 'control'))+' | '+' | '.join(str(s['counts'][k]) for k in kinds)+' |')
    lines += ['', 'Each row contains 72 dependent seed/locality views; these counts are not independent replications.',
        'Post-freeze source-development diagnosis only. No trajectory, deployment, metric/seconds or physical-safety claim.']
    (run.PUBLIC/'error_accounting.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps([s for s in summaries if s['variant'] == 'oof' and s['arm'] == 'risk_context'], indent=2))


if __name__ == '__main__': main()
