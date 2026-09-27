"""Descriptive fitting-only support, slope and easy-cut drift after fixed fits."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_oof_magnitude as run
import numpy as np


def summary(values):
    a = np.asarray(values, float)
    if not len(a) or not np.isfinite(a).all(): raise ValueError('Finite completed values required')
    return dict(count=len(a), minimum=float(a.min()), median=float(np.median(a)), maximum=float(a.max()))


def main():
    run.check_freeze()
    frozen = json.loads((run.PUBLIC/'inner_prediction_freeze.json').read_text())
    inner = [json.loads((ROOT/r['path']).read_text()) for r in frozen['inner_heads']]
    models = [json.loads((ROOT/r['path']).with_name('models.json').read_text())
        for r in json.loads((run.PUBLIC/'prediction_freeze.json').read_text())['receipts']]
    result = dict(result_source='fresh_run_fitting_diagnostic_not_new_readout', by_family={}, cut_drift=[], training_loss={})
    for arm in ('cap_aux', 'shuffled_aux'):
        selected = [r for r in inner if r['identity']['arm'] == arm]
        first = [r['fit']['trace'][0]['cost_loss'] for r in selected]
        last = [r['fit']['trace'][-1]['cost_loss'] for r in selected]
        result['training_loss'][arm] = dict(first=summary(first), final=summary(last),
            decreasing_heads=sum(b < a for a,b in zip(first, last)),
            interpretation='fixed_batch_scaled_cost_loss_not_held_prediction_quality')
    for family in ('full', 'motion_only'):
        selected = [r for r in models if r['details']['cost_only']['outer_original']['path'].split('/')[-3].find('_'+family+'_') >= 0]
        result['by_family'][family] = {}
        for arm in ('cost_only', 'cap_aux', 'shuffled_aux'):
            slopes = np.asarray([r['models'][arm]['slopes'] for r in selected])
            result['by_family'][family][arm] = dict(H_all=summary(slopes[:, 0]), H_easy=summary(slopes[:, 1]),
                lower_bound_components=int((slopes == 0).sum()), upper_bound_components=int((slopes == 8).sum()))
        events = [r for r in inner if r['identity']['arm'] == 'cap_aux'
            and '_'+family+'_' in r['identity']['input']['original']['tag']]
        result['by_family'][family]['inner_cap_events'] = dict(prior=summary([r['fit']['cap_prior'] for r in events]),
            zero_positive_fits=sum(r['identity']['input']['event_positive'] == 0 for r in events),
            positive_rows=summary([r['identity']['input']['event_positive'] for r in events]))
    result['cut_drift'] = [dict(outer=r['outer'], training_sites=r['fitting_sites'],
        **r['details']['cost_only']['cut_drift']) for r in models]
    run.immutable_json(run.PUBLIC/'fitting_diagnostics.json', result)
    lines = ['# Fitting Magnitude and Target-Definition Drift', '',
        'Descriptive only. No slope,cut,model or deployment policy was selected from these summaries.', '',
        '| Family/arm | H_all slope min/median/max | H_easy slope min/median/max | Bound components low/high |', '|---|---|---|---|']
    for family, arms in result['by_family'].items():
        for arm in ('cost_only', 'cap_aux', 'shuffled_aux'):
            row = arms[arm]
            a = [row['H_all'][k] for k in ('minimum', 'median', 'maximum')]
            b = [row['H_easy'][k] for k in ('minimum', 'median', 'maximum')]
            lines.append(f"| {family}/{arm} | {a} | {b} | {row['lower_bound_components']}/{row['upper_bound_components']} |")
        lines += ['', f"{family} inner event support: `{json.dumps(arms['inner_cap_events'])}`", '']
    lines += ['The JSON retains every outer-view cut drift. The readout is trained against producer-specific',
        'two-locality easy definitions and applied to three-locality outer heads. Single-locality cap-reference',
        'training is a further support/transport limitation. None of these diagnostics proves causation.',
        'Obs8/pred12 native steps,detector pixels,source development only. Stage5C/SMC off.']
    (run.PUBLIC/'fitting_diagnostics.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__': main()
