"""All prespecified cost-shape comparisons, including negative intervals."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_cost_shape as run
from scripts.report_m3w_european_support_fractional import paired_contrasts
from scripts.report_m3w_european_frozen_harm_readout import summarize_contrasts
from scripts.report_m3w_european_cap_auxiliary_cost import PRIMARY,GUARDS
from scripts.report_m3w_european_cost_mass import summarize

COMPARISONS={f'shape_mass_{arm}_vs_{mode}_{arm}':(arm+'_shape_mass',arm+'_'+mode)
    for arm in ('cost_only','cap_aux','shuffled_aux') for mode in ('raw','scaled','mass','shape_L2')}
COMPARISONS.update({f'shape_L2_{arm}_vs_scaled_{arm}':(arm+'_shape_L2',arm+'_scaled')
    for arm in ('cost_only','cap_aux','shuffled_aux')})
COMPARISONS.update(shape_true_vs_shape_cost=('cap_aux_shape_mass','cost_only_shape_mass'),
    shape_true_vs_shape_shuffled=('cap_aux_shape_mass','shuffled_aux_shape_mass'))


def aggregate(rows,cfg,supported):
    contrasts={}
    for key,(new,old) in COMPARISONS.items():
        proxy=[dict(row,folds=[dict(f,metrics=dict(fractional=f['metrics'][new],mean=f['metrics'][old]))
            for f in row['folds']]) for row in rows]
        contrasts[key]=paired_contrasts(proxy,cfg)
    summary=summarize_contrasts(contrasts)
    def screen(names):
        primary=all(summary[n]['full'][PRIMARY]['positive']==6 for n in names)
        guards=all(summary[n]['full'][g]['negative']==summary[n]['full'][g]['not_estimable']==0 for n in names for g in GUARDS)
        return primary,guards
    primary,guards=screen(['shape_mass_cost_only_vs_raw_cost_only','shape_mass_cost_only_vs_scaled_cost_only'])
    info,info_guards=screen(['shape_true_vs_shape_cost','shape_true_vs_shape_shuffled'])
    return dict(contrasts=contrasts,summary=summary,gates=dict(
        fitting_shape_and_mass_supported=supported,primary_readout_gate=primary,guard_gate=guards,
        auxiliary_information_gate=info and info_guards,development_readout_screen=supported and primary and guards,
        independent_confirmation=False,new_policy_evaluated=False,deployment_changed=False,
        submission_ready=False,stage5c_executed=False,smc_enabled=False))


def fitting_summary(records):
    result={}
    for family in ('full','motion_only'):
        rows=[r for r in records if r['pair']==family]; assert len(rows)==72
        result[family]={}
        for arm in ('cost_only','cap_aux','shuffled_aux'):
            for mode in ('shape_L2','shape_mass'):
                key=arm+'_'+mode; models=[r['models'][key] for r in rows]
                result[family][key]=dict(views=72,mass_preserved=sum(int(m['mass_preserved']) for m in models),
                    components={label:{metric:summarize([m['components'][j][metric] for m in models])
                        for metric in ('MSE','target_mass','predicted_mass','normalized_optimality_gap','positive_increments')}
                        for j,label in enumerate(('H_all','H_easy'))})
    return result


def main():
    cfg,_=run.registration(); freeze=run.check_freeze()
    records=[json.loads((ROOT/ref['path']).with_name('models.json').read_text()) for ref in freeze['readouts']]
    fs=fitting_summary(records); run.immutable_json(run.PUBLIC/'fitting_diagnostics.json',fs)
    supported=all(v['mass_preserved']==72 for arms in fs.values() for name,v in arms.items() if name.endswith('shape_mass'))
    rows=json.loads((run.PUBLIC/'readout.json').read_text())['rows']; out=aggregate(rows,cfg,supported)
    run.immutable_json(run.PUBLIC/'aggregate_metrics.json',out)
    run.immutable_json(run.PUBLIC/'compute_receipt.json',dict(fresh_shape_readouts=864,
        knot_ordinates_per_readout=10,component_quadratic_programs=1728,new_neural_heads=0,new_optimizer_updates=0,
        cached_inner_neural_heads=1296,cached_outer_neural_heads=432,source_held_views=144,
        direct_MSE_checks=1728,threads=4,interop=1,workers=0,remote_jobs_submitted=0))
    lines=['# Monotone Cost Shape Results','','## Material Passport',
        'fresh_run:864 small monotone readouts and144 source-held views; no new neural training.',
        'cached_verified:honest nested OOF neural scores,forecasts,features and raw/L2/mass controls.',
        'not_run:new policy,independent selection,reserved calibration,confirmation and deployment.','',
        'Obs8/pred12 native steps;detector pixels. No seconds/metric/human-gold/true3D/foundation claim.',
        'Three seeds average within locality;3000 paired resamples of four localities per assignment.',
        'Six overlapping exposed-source assignments are descriptive;intervals unadjusted.','',
        '| Contrast/family | Positive/negative/overlap/missing primary CIs | Point range (%) |','|---|---|---|']
    for name,families in out['summary'].items():
        for family,metrics in families.items():
            v=metrics[PRIMARY]; lines.append(f"| {name}/{family} | {[v[k] for k in ('positive','negative','overlap','not_estimable')]} | {v['point_range']} |")
    lines+=['','## Every Primary Interval','','| Contrast/family/assignment | Point (%) | 95% CI |','|---|---:|---|']
    for name,families in out['contrasts'].items():
        for family,groups in families.items():
            for group,metrics in groups.items():
                v=metrics[PRIMARY]; lines.append(f"| {name}/{family}/{group} | {v.get('point')} | {v.get('CI','not_estimable')} |")
    lines+=['','## Boundaries',
        'Primary is expected easy-harm MSE,not ADE/FDE or easy-degradation. Guards are unchanged.',
        'Sequential H_all then H_easy fits are not joint MSE optimization. Easy caps differ across modes.',
        'Training mean constraints are not conditional or held-locality calibration guarantees.',
        'Both modes use raw H_all/envelope and H_easy/H_all fractions;shape and nested scaling change together relative to the old origin readout.',
        'Full/motion forecast populations differ;their contrast is not a matched causal feature ablation.',
        '```json',json.dumps(out['gates'],indent=2),'```']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(gates=out['gates'],primary={k:v['full'][PRIMARY] for k,v in out['summary'].items()}),indent=2))


if __name__=='__main__': main()
