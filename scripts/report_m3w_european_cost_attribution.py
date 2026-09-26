"""All source assignments, including adverse label-assisted comparisons."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_cost_attribution as run
import numpy as np
METRIC_NAMES=('composed_gain','E_assisted_gain','E_assisted_gain_vs_composed','H_assisted_gain',
    'membership_over_MSE','severity_over_MSE','cross_over_MSE','Brier','severity_weighted_Brier',
    'outside_easy_share','outside_easy_low_p_share','outside_easy_high_severity_share','all_harm_E_assisted_gain')


def metrics(fold,subset):
    d=fold['diagnosis']['subsets'][subset]
    if d.get('status')=='not_estimable': return None
    e=d['easy']; a=d['all_harm']; mse=e['MSE']
    ratio=lambda v:v/mse if mse>0 else None
    return dict(composed_gain=e['gain_vs_original_percent'],
        E_assisted_gain=e['label_assisted_E_gain_vs_original_percent'],
        E_assisted_gain_vs_composed=e['label_assisted_E_gain_vs_composed_percent'],
        H_assisted_gain=e['label_assisted_H_gain_vs_original_percent'],
        membership_over_MSE=ratio(e['membership_squared']),
        severity_over_MSE=ratio(e['severity_squared']),cross_over_MSE=ratio(e['cross_term']),
        Brier=d['membership_Brier'],severity_weighted_Brier=d['severity_weighted_Brier'],
        outside_easy_share=d['strata']['outside_easy']['easy_MSE_share'],
        outside_easy_low_p_share=d['strata']['outside_easy_low_probability']['easy_MSE_share'],
        outside_easy_high_severity_share=d['strata']['outside_easy_high_severity']['easy_MSE_share'],
        all_harm_E_assisted_gain=a['label_assisted_E_gain_vs_original_percent'])


def aggregate(rows):
    out={}; counts={}
    for pair in ('full','motion_only'):
        out[pair]={}; counts[pair]={}
        for subset in ('all','disagreement'):
            out[pair][subset]={}
            for a,b in sorted({(r['producer'],r['controller']) for r in rows}):
                rs=[r for r in rows if r['pair']==pair and (r['producer'],r['controller'])==(a,b)]
                assert len(rs)==3 and sorted(r['seed'] for r in rs)==[17,29,43]
                sites=sorted({f['held'] for r in rs for f in r['folds']}); assert len(sites)==4
                vals={s:[metrics(next(f for f in r['folds'] if f['held']==s),subset) for r in rs] for s in sites}
                group={}
                for key in METRIC_NAMES:
                    values=[]
                    for site in sites:
                        vs=[v[key] if v is not None else None for v in vals[site]]
                        if any(v is None or not np.isfinite(v) for v in vs): break
                        values.append(float(np.mean(vs)))
                    if len(values)!=4: group[key]=dict(status='not_estimable'); continue
                    rng=np.random.default_rng(47131); v=np.array(values)
                    draws=v[rng.integers(0,4,(3000,4))].mean(1)
                    group[key]=dict(point=float(v.mean()),CI=np.quantile(draws,[.025,.975]).tolist(),locality_points=dict(zip(sites,values)))
                out[pair][subset][f'producer{a}_controller{b}']=group
            counts[pair][subset]={}
            for key in next(iter(out[pair][subset].values())):
                vs=[v[key] for v in out[pair][subset].values()]; good=[v for v in vs if 'CI' in v]
                counts[pair][subset][key]=dict(positive=sum(v['CI'][0]>0 for v in good),
                    negative=sum(v['CI'][1]<0 for v in good),overlap=sum(v['CI'][0]<=0<=v['CI'][1] for v in good),
                    not_estimable=len(vs)-len(good),point_range=[min(v['point'] for v in good),max(v['point'] for v in good)] if good else None)
    return dict(contrasts=out,summary=counts,label_assisted_not_model_results=True)


def main():
    identity=run.registration(); done=json.loads((run.PUBLIC/'completion_checks.json').read_text()); assert done['identity']==identity
    rows=[]
    for ref in done['groups']:
        assert run.artifact(ROOT/ref['path'])==ref; rows.append(json.loads((ROOT/ref['path']).read_text()))
    result=aggregate(rows); run.immutable_json(run.PUBLIC/'aggregate_metrics.json',result)
    lines=['# Frozen Factor Attribution','','## Material Passport',
        'Fresh offline attribution from cached_verified models. No new fitting, policy, threshold or independent evaluation.',
        'Label-assisted substitutions require future-derived labels and are NOT deployable models.','',
        '| Pair / source roles | Current cost gain % | E-assisted gain % | H-assisted gain % | E-assisted 95% CI |',
        '|---|---:|---:|---:|---:|']
    for pair,ss in result['contrasts'].items():
        for role,d in ss['disagreement'].items():
            fmt=lambda k:'not_estimable' if 'point' not in d[k] else f"{d[k]['point']:.3f}"
            lines.append(f"| {pair} / {role} | {fmt('composed_gain')} | {fmt('E_assisted_gain')} | {fmt('H_assisted_gain')} | {d['E_assisted_gain'].get('CI','not_estimable')} |")
    lines+=['','Gains compare easy-harm MSE to the original cost model, not ADE/FDE.',
        'Three seeds averaged per locality; 3,000 resamples of four localities. Repeated source roles/windows are dependent and exploratory.',
        'The signed cross term can be negative. Squared-term/MSE ratios are not independent causal shares or probabilities.',
        'No change to the parent failed model gate. Pixels/annotation steps; no metric/seconds, gold, physical-safety, true3D or foundation claim. Stage5C/SMC off.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result['summary']['full']['disagreement'],indent=2))


if __name__=='__main__': main()
