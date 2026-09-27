"""Matched descriptor readout, including negative and undefined risk outcomes."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_causal_descriptor_refit as run
from scripts.report_m3w_fixed_floor_tail import ci


def main():
    d = json.loads((run.PUBLIC/'summary.json').read_text())
    lines = ['# Causal-Descriptor Risk-Head Results', '', '## Material Passport', '',
        'Fresh108risk heads,216,000updates. Cached_verified108signed-excess controls, original forecasting/floor/utility chain.',
        'Twelve opened development localities, three forecaster seeds. Independent roles stay closed.',
        'Six causal features and384zero-initialized branch parameters added; no loss, budget, draw or deployment change.', '',
        '## Prespecified Primary', '',
        'Count-matched control minus new selected harm (pp): **'+ci(d['paired']['control_matched_count']['positive_harm_reduction_pp'])+'**.',
        'Equal-current-query-count ADE advantage (%): **'+ci(d['paired']['control_matched_count']['ADE_gain_percent'])+'**.',
        'Undefined fixed-roster risk is not zero risk. Count matching is diagnostic, not a certified2%policy.', '',
        '| Policy | ADE / floor gain % | Hard / floor gain % | Easy / CV gain % | FDE / floor gain % | Intervention % | Selected harm % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p, m in d['summary'].items():
        lines.append('| '+p+' | '+' | '.join(ci(m[k]) for k in ('all_gain_floor','hard_gain_floor','easy_gain_CV','FDE_gain_floor'))+' | '+ci(m['intervention_rate'],100)+' | '+ci(m['selected_positive_harm_ratio'],100)+' |')
    lines += ['', '| New versus control | ADE gain % | Harm reduction pp | Intervention difference pp |', '|---|---:|---:|---:|']
    for p, m in d['paired'].items(): lines.append('| '+p+' | '+' | '.join(ci(m[k]) for k in ('ADE_gain_percent','positive_harm_reduction_pp','intervention_difference_pp'))+' |')
    lines += ['', '## Worst Views and Label Sensitivity', '',
        '| Policy | Worst easy gain % | Risk violations /216 | Undefined views | Complete ADE gain % | Partial ADE gain % | Unknown interventions /view |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p, m in d['summary'].items():
        w = d['worst_views'][p]
        lines.append(f"| {p} | {w['worst_easy_gain_CV']:.4f} | {w['risk_violating_views']} | {w['undefined_risk_views']} | "+' | '.join(ci(m[k]) for k in ('complete_gain_floor','partial_gain_floor','unknown_interventions'))+' |')
    lines += ['', '## Training Fit Versus Held Error', '',
        '| Arm | Fitting all MSE | Held all MSE | Fitting easy MSE | Held easy MSE |', '|---|---:|---:|---:|---:|']
    for p, m in d['quality'].items(): lines.append('| '+p+' | '+' | '.join(ci(m[k]) for k in ('all_fit_MSE','all_signed_MSE','easy_fit_MSE','easy_signed_MSE'))+' |')
    lines += ['', 'Scores divide by fitting-only cost scale. In-sample fit is not validation; score components are not identified cost moments.', '',
        '## Three Forecaster Seeds', '', '| Seed | New ADE/floor gain % | Control ADE/floor gain % | New harm % |', '|---|---:|---:|---:|']
    for seed, m in d['by_seed'].items(): lines.append('| '+seed+' | '+ci(m['descriptor']['all_gain_floor'])+' | '+ci(m['control']['all_gain_floor'])+' | '+ci(m['descriptor']['selected_positive_harm_ratio'],100)+' |')
    lines += ['', '## Gates', '', *[f'- {k}: {v}' for k,v in d['gates'].items()], '',
        'Bootstrap3,000draws over12locality means after dependent producer/fit/seed averaging. Overlapping windows are not independent.',
        'Per-source values are in summary.json. No source is dropped to rescue undefined risk. Development intervals are not safety certificates.',
        'This tests six extra descriptors/384parameters together, not their individual semantic value under equal capacity.',
        'Image-local detector silver; obs8/pred12 rawstride12. No metric/seconds/physical-safety/true3D/foundation claim.',
        'No independent confirmation, deployment, Stage5C or SMC.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    freeze = json.loads((run.PUBLIC/'decision_freeze.json').read_text()); infos = []; weights = []
    for ref in freeze['heads']:
        assert run.base.artifact(ROOT/ref['path']) == ref
        rec = json.loads((ROOT/ref['path']).read_text()); infos.append(rec['fit'])
        weights.append(float(run.api.read_checkpoint(ROOT/rec['artifacts']['checkpoint']['path'])['model']['descriptor.weight'].norm()))
    training = dict(new_heads=len(infos), updates=sum(r['step'] for r in infos), fit_seconds=sum(r['seconds'] for r in infos),
        nonzero_descriptor_branches=sum(w>0 for w in weights), unknown_supervised_draws=sum(r['unknown_rows_sampled'] for r in infos),
        declining_fixed_loss_heads=sum(r['trace'][-1]['fixed_training_excess_MSE']<r['trace'][0]['fixed_training_excess_MSE'] for r in infos),
        mean_initial_fixed_loss=float(np.mean([r['trace'][0]['fixed_training_excess_MSE'] for r in infos])),
        mean_final_fixed_loss=float(np.mean([r['trace'][-1]['fixed_training_excess_MSE'] for r in infos])),
        parameters_per_head=infos[0]['parameters'], additional_parameters=384)
    run.base.immutable_json(run.PUBLIC/'training_summary.json', training)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['Signed-excess control','Causal descriptors','Control: matched count']; policies = ['control','descriptor','control_matched_count']
    fig, axes = plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,key,mult,title in ((axes[0],'all_gain_floor',1,'Incremental ADE gain (%)'),(axes[1],'selected_positive_harm_ratio',100,'Selected positive harm (%)')):
        for i,p in enumerate(policies):
            v = d['summary'][p][key]
            if v['point'] is None: ax.text(.05,i,'undefined',transform=ax.get_yaxis_transform(),va='center'); continue
            pt=v['point']*mult; lo,hi=np.asarray(v['ci95'])*mult
            ax.errorbar(pt,i,xerr=[[pt-lo],[hi-pt]],fmt='o',capsize=3,color='#167d73' if p=='descriptor' else '#555555')
        ax.set_yticks(range(3),labels); ax.set_ylim(2.5,-.5); ax.set_title(title)
        ax.axvline(0 if key=='all_gain_floor' else 2,color='#ba4340',linestyle='--',linewidth=1); ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Matched causal-descriptor refit: development evidence only')
    fig.savefig(run.PUBLIC/'risk_gain_comparison.png',dpi=150,metadata={'Software':'M3W causal descriptor refit'}); plt.close(fig)
    print(json.dumps(dict(training=training,gates=d['gates'],primary=d['paired']['control_matched_count'])))


if __name__ == '__main__': main()
