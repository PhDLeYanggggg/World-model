"""Prespecified source-site contrasts and conditional blocked uncertainty."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.run_m3w_source_site_probe import SiteCorpus, load_config
from scripts.analyze_m3w_source_visual_start import paired_brier_parts, leave_one_agent_sensitivity
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_motion_start_probe import paired_agent_interval
from src.world_model.m3w_offline_visual_data import json_write


def paired_block_interval(y, p, reference, groups, *, resamples=2000, seed=421):
    y, p, reference, groups = map(np.asarray, (y,p,reference,groups))
    if (y.ndim != 1 or p.ndim != 2 or p.shape != reference.shape or p.shape[1] != len(y)
            or groups.shape != y.shape or not len(y) or not np.isfinite(p).all()
            or not np.isfinite(reference).all() or resamples < 1):
        raise ValueError('Aligned finite seed-by-row predictions and blocks required')
    # Average losses across seeds before resampling: this is not an ensemble.
    gains = ((reference-y)**2-(p-y)**2).mean(0)
    unique, inverse = np.unique(groups,return_inverse=True)
    sums = np.bincount(inverse,weights=gains)
    counts = np.bincount(inverse)
    if len(unique) < 2:
        return dict(row_lift=float(gains.mean()), blocks=len(unique), available=False)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0,len(unique),size=(resamples,len(unique)))
    boot = sums[sampled].sum(1)/counts[sampled].sum(1)
    return dict(row_lift=float(gains.mean()),blocks=len(unique),available=True,
        conditional_ci95=np.quantile(boot,[.025,.975]).tolist(), resamples=resamples,
        overlapping_training_folds=True,independent_confirmation=False)


def equal_site_interval(site_values, *, resamples=2000, seed=422):
    values = np.asarray(site_values,dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Finite distinct physical-site contrasts required')
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0,len(values),(resamples,len(values)))].mean(1)
    return dict(equal_site_lift=float(values.mean()),physical_sites=len(values),
        conditional_ci95=np.quantile(boot,[.025,.975]).tolist(),resamples=resamples,
        overlapping_training_folds=True,independent_confirmation=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    args = parser.parse_args(); reg = load_config(args.registration)
    reports, out = ROOT/reg['reports'], ROOT/reg['output']
    rp = reports/'report.json'; report = json.loads(rp.read_text())
    if report['completed_models'] != 30 or report['optimizer_updates'] != 60000:
        raise ValueError('Full fixed matrix required before analysis')
    data = SiteCorpus(reg)
    if data.assignment_hash != report['identity']['source_assignment_sha256']:
        raise ValueError('Source row alignment changed')
    lookup = {(t['arm'],t['site'],t['seed']):t for t in report['trials']}
    predictions, held_ids = {}, {}
    for t in report['trials']:
        if file_digest(ROOT/t['prediction_path']) != t['prediction_sha256']:
            raise ValueError('Prediction integrity failure')
        with np.load(ROOT/t['prediction_path'],allow_pickle=False) as a:
            predictions[t['trial']] = a['probability'].copy()
            held_ids[t['trial']] = a['held_indices'].copy()
    rows, losses, sites, video_rows = [], [], [], []
    for site in reg['sites']:
        _,weights,held,_ = data.source_design(site)
        local = held-data.nmain; y = data.y[held]
        ps, masks, priors = [], [], []
        parts, own_parts = [], []
        for seed in reg['seeds']:
            control = lookup['mask_only',site,seed]
            mask = predictions[control['trial']]
            for arm in reg['arms']:
                t = lookup[arm,site,seed]; p = predictions[t['trial']]
                np.testing.assert_array_equal(held_ids[t['trial']],held)
                prior = np.full(len(y),t['training_prior'])
                part = paired_brier_parts(y,p,mask); own = paired_brier_parts(y,p,prior)
                e = t['evaluation']
                row = dict(trial=t['trial'],site=site,seed=seed,arm=arm,rows=len(y),
                    brier=e['brier'],mask_lift=part['total_lift'],own_prior_lift=own['total_lift'],
                    auroc=e['auroc'],auprc=e['auprc'],ece=e['ece'],log_loss=e['log_loss'],
                    mean_probability=float(p.mean()),positive_rate=float(y.mean()),
                    training_prior=t['training_prior'],training_brier=t['training_weighted_brier'],
                    mask_mean_shift=part['mean_shift_contribution'],mask_varying_prediction=part['within_site_variation_contribution'],
                    prior_mean_shift=own['mean_shift_contribution'],prior_varying_prediction=own['within_site_variation_contribution'])
                rows.append(row)
                losses.extend(dict(trial=t['trial'],**loss) for loss in t['fit']['losses'])
                if arm == 'past_rgb':
                    ps.append(p); masks.append(mask); priors.append(prior); parts.append(part); own_parts.append(own)
        ps,masks,priors = map(np.asarray,(ps,masks,priors))
        records,agents = data.source_records[local],data.source_tracks[local]
        block = paired_block_interval(y,ps,masks,records,resamples=reg['bootstrap_resamples'])
        prior_block = paired_block_interval(y,ps,priors,records,resamples=reg['bootstrap_resamples'])
        rgb_rows = [r for r in rows if r['site'] == site and r['arm'] == 'past_rgb']
        site_result = dict(site=site,rows=len(y),videos=len(set(records)),agents=len(set(agents)),
            means={k:float(np.mean([r[k] for r in rgb_rows])) for k in
                ('brier','mask_lift','own_prior_lift','auroc','auprc','ece','log_loss','training_brier',
                 'mask_mean_shift','mask_varying_prediction','prior_mean_shift','prior_varying_prediction')},
            positive_rgb_mask_seeds=sum(r['mask_lift']>0 for r in rgb_rows),
            mask_video_interval=block,own_prior_video_interval=prior_block,
            mask_agent_interval=paired_agent_interval(y,ps,masks,agents),
            mask_agent_omission=leave_one_agent_sensitivity(y,ps,masks,agents))
        sites.append(site_result)
        for recording in sorted(set(records)):
            m = records == recording
            video_rows.append(dict(site=site,recording=recording,rows=int(m.sum()),agents=len(set(agents[m])),
                positive_rate=float(y[m].mean()),
                mask_lift=float((((masks[:,m]-y[m])**2)-((ps[:,m]-y[m])**2)).mean()),
                own_prior_lift=float((((priors[:,m]-y[m])**2)-((ps[:,m]-y[m])**2)).mean())))
    primary = equal_site_interval([s['means']['mask_lift'] for s in sites],resamples=reg['bootstrap_resamples'])
    own = equal_site_interval([s['means']['own_prior_lift'] for s in sites],resamples=reg['bootstrap_resamples'])
    agent = equal_site_interval([s['mask_agent_interval']['agent_balanced_lift'] for s in sites],resamples=reg['bootstrap_resamples'])
    result = dict(result_source='fresh_run_analysis_of_fixed_cached_verified_predictions',
        report_sha256=file_digest(rp),analysis_script_sha256=file_digest(Path(__file__)),
        primary=primary,own_training_prior=own,equal_agent_secondary=agent,sites=sites,
        positive_row_sites=sum(s['means']['mask_lift']>0 for s in sites),
        positive_agent_sites=sum(s['mask_agent_interval']['agent_balanced_lift']>0 for s in sites),
        positive_prior_sites=sum(s['means']['own_prior_lift']>0 for s in sites),
        seed_losses_averaged=True,models_or_thresholds_selected=False,
        main_primary_changed=False,sealed_roles_opened=False,new_forecasting_lift=False,
        independent_confirmation=False,new_deployment=False,stage5c_executed=False,smc_enabled=False)
    json_write(reports/'analysis.json',result)
    for filename,items in [('fit_metrics.csv',rows),('loss_trace.csv',losses),('video_metrics.csv',video_rows)]:
        with (reports/filename).open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(items[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(items)
    ci = primary['conditional_ci95']
    lines = ['# Source-Site Visual Information Results','',
        'Internal held-physical-site diagnostic of original SDD train40 only. No formal main split changed.',
        'All five sites and three seeds are retained. Average seed losses, not ensemble predictions.',
        'Positive absolute Brier reduction is not percentage ADE/FDE improvement.','',
        f"Primary equal-site RGB-minus-mask lift: {primary['equal_site_lift']:+.6f}; conditional site-block interval [{ci[0]:+.6f}, {ci[1]:+.6f}].",
        'Only five physical sites, overlapping training folds and already exposed source data: not independent confirmation.', '',
        '| Held site | Windows / agents / videos | RGB Brier | Lift vs mask | Video-block conditional95% interval | Lift vs training prior | Positive seeds |',
        '| --- | --- | ---: | ---: | --- | ---: | ---: |']
    for s in sites:
        m=s['means'];lo,hi=s['mask_video_interval']['conditional_ci95']
        lines.append(f"| {s['site']} | {s['rows']} / {s['agents']} / {s['videos']} | {m['brier']:.6f} | {m['mask_lift']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {m['own_prior_lift']:+.6f} | {s['positive_rgb_mask_seeds']}/3 |")
    lines += ['','## Weighting and Prior Sensitivity','',
        '| Site | Equal-agent RGB lift [conditional95% interval] | Mask mean-shift term | Mask varying-prediction term | Prior varying-prediction term | RGB AUROC | RGB ECE |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for s in sites:
        a=s['mask_agent_interval']; m=s['means'];lo,hi=a['descriptive_ci95']
        lines.append(f"| {s['site']} | {a['agent_balanced_lift']:+.6f} [{lo:+.6f}, {hi:+.6f}] | {m['mask_mean_shift']:+.6f} | {m['mask_varying_prediction']:+.6f} | {m['prior_varying_prediction']:+.6f} | {m['auroc']:.6f} | {m['ece']:.6f} |")
    lines += ['',f"Equal-site training-prior lift: {own['equal_site_lift']:+.6f}; equal-site/equal-agent mask lift: {agent['equal_site_lift']:+.6f}.",
        'Agent-weighted contrasts answer a different question and do not replace the primary window-within-site contrast.',
        'Brier decomposition uses held labels for fixed-result diagnosis only; no held-label calibration is applied.',
        'The target is any annotation-coordinate change, not human motion intention. Incomplete future labels remain unscored.',
        'SDD8-to12 at stride12 is +144rawframes, not physically time-equated to main. No metric/seconds claim.',
        'No forecast policy, model promotion, Stage5C or SMC.','']
    (reports/'results.md').write_text('\n'.join(lines))
    cache=out/'plot_runtime';cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache/'matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME',str(cache/'xdg'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt']='m3w-source-site-probe-v1'
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,key,title in zip(axes,['mask_lift','own_prior_lift'],['RGB vs matched mask','RGB vs training constant prior']):
        values=[s['means'][key] for s in sites]
        ax.barh(range(len(sites)),values,color=['#238274' if v>0 else '#b74751' for v in values],height=.6)
        for i,s in enumerate(sites):
            pts=[r[key] for r in rows if r['arm']=='past_rgb' and r['site']==s['site']]
            ax.scatter(pts,[i]*len(pts),c='black',s=14,zorder=3)
        ax.axvline(0,color='black',lw=.8);ax.grid(axis='x',alpha=.2)
        ax.spines[['top','right']].set_visible(False);ax.set_title(title);ax.set_xlabel('Absolute Brier reduction')
    axes[0].set_yticks(range(len(sites)),[s['site'] for s in sites]);axes[0].invert_yaxis()
    fig.suptitle('Source-internal held-site diagnosis: fixed three-seed comparisons')
    fig.text(.5,.01,'Bars average seed losses; points are seeds. Exposed source fit data, not independent confirmation.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.95))
    fig.savefig(reports/'source_site_lift.svg',metadata={'Date':None})
    fig.savefig(cache/'source_site_lift.png',dpi=130);plt.close(fig)
    svg=reports/'source_site_lift.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='sites'},indent=2))


if __name__=='__main__':
    main()
