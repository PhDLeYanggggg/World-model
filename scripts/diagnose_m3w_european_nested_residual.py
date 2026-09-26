"""Post-freeze descriptive error accounting; never a fitting or selection rule."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_european_nested_residual as run


def accounting(original, corrected, target, use):
    a,b,y = [np.asarray(x,float) for x in (original,corrected,target)]
    use = np.asarray(use,bool) & np.isfinite(a) & np.isfinite(b) & np.isfinite(y)
    if not use.any(): raise ValueError('Known rows required')
    e,d = a[use]-y[use],b[use]-a[use]
    old=float(np.mean(e**2)); new=float(np.mean((e+d)**2))
    cross=float(2*np.mean(e*d)); energy=float(np.mean(d**2))
    np.testing.assert_allclose(new-old,cross+energy,rtol=1e-9,atol=1e-10)
    reason = ('improved' if new<old else 'unchanged' if new==old else
              'wrong_aggregate_direction' if cross>=0 else 'useful_direction_excess_magnitude')
    return dict(rows=int(use.sum()),original_MSE=old,corrected_MSE=new,cross_term=cross,
                shift_energy=energy,MSE_change=new-old,description=reason,
                positive_shift_fraction=float(np.mean(d>0)),negative_shift_fraction=float(np.mean(d<0)),
                mean_shift=float(d.mean()),mean_error_before=float(e.mean()))


def main():
    cfg,identity=run.registration()
    completion=json.loads((run.PUBLIC/'completion_checks.json').read_text()); assert completion['all_passed']
    run.base.previous.require_committed(run.PUBLIC/'prediction_freeze.json')
    rows=[]
    for g,data,pairs in run.base.contexts(identity['source']):
        bi=pairs['B']['ids']; sites=data['sites'][bi]; cv=data['baseline_ade'][bi,1]
        for pair in cfg['pairs']:
            _,env,by,*_=run.base.pair_inputs(g,data,pairs,pair)
            result=json.loads((run.PUBLIC/'groups'/(g['group']+'_'+pair+'.json')).read_text())
            original=json.loads((run.source.PUBLIC/'groups'/(g['group']+'_'+pair+'.json')).read_text())
            for held in sorted(set(sites)):
                te=sites==held; tag=g['group']+'_'+pair+'_'+held
                prior=next(f for f in original['folds'] if f['held']==held)
                y=run.source.tail.diagnostic.event_targets(by[te],cv[te],prior['easy_cut'])
                f=next(f for f in result['folds'] if f['held']==held)
                assert run.array_hash(y)==f['target_sha256']
                with np.load(run.parent.frozen_directory(tag,'original')/'scores.npz',allow_pickle=False) as z: p=z['scores'].copy()
                with np.load(run.PRIVATE/'probes'/tag/'scores.npz',allow_pickle=False) as z:
                    for variant in cfg['residual_variants']:
                        key=variant+'__context_bias'; q=z[key]
                        a=accounting(p[:,3],q[:,3],y[:,3],env[te]>0)
                        np.testing.assert_allclose(a['corrected_MSE'],f['metrics'][key]['envelope_positive']['harm_MSE'],rtol=1e-12,atol=1e-12)
                        rows.append(dict(tag=tag,pair=pair,variant=variant,held=held,**a,
                            easy_cut_label_disagreement=f['cut_drift'][variant]['easy_label_disagreement_fraction']))
    assert len(rows)==432
    summary={}
    for pair in cfg['pairs']:
        summary[pair]={}
        for variant in cfg['residual_variants']:
            rr=[r for r in rows if r['pair']==pair and r['variant']==variant]
            kinds=['improved','unchanged','wrong_aggregate_direction','useful_direction_excess_magnitude']
            summary[pair][variant]={k:sum(r['description']==k for r in rr) for k in kinds}
    doc=dict(result_source='fresh_run_post_freeze_descriptive_accounting',rows=rows,summary=summary,
             predictions_changed=False,held_guided_model_selection=False,causal_root_cause_proven=False,
             independent_views=False,new_hyperparameters=False)
    run.immutable_json(run.PUBLIC/'residual_error_accounting.json',doc)
    lines=['# Frozen Residual Error Accounting','','Post-freeze descriptive diagnosis, not a preregistered primary gate or new model.',
        'For original error e and applied shift d, MSE change = 2 mean(e*d) + mean(d^2).',
        'The cross term measures aggregate alignment; the nonnegative shift energy can overwhelm a helpful direction.',
        'Wrong aggregate direction includes a zero, non-helpful cross term; it is not a per-row direction claim.',
        'This algebra is not a causal explanation and does not identify a future-safe shrinkage factor. No factors are fitted.',
        '', '| Inputs / residual bank | Improved | Unchanged | Wrong aggregate direction | Useful direction, excess magnitude |',
        '|---|---:|---:|---:|---:|']
    for pair,vv in summary.items():
        for variant,s in vv.items(): lines.append('| '+pair+' / '+variant+' | '+' | '.join(str(s[k]) for k in kinds)+' |')
    lines+=['','Each row summarizes72 dependent seed/locality views; counts are not independent evidence.',
        'All432 decompositions match direct MSE arithmetic. No held-optimal correction coefficient is used for fitting or deployment.',
        'Eight observed/twelve predicted annotation steps, image pixels only. Not metric/seconds, physical safety, true3D or foundation evidence.']
    (run.PUBLIC/'residual_error_accounting.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
