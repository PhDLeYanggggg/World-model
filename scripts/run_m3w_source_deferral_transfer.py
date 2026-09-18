"""Read-only frozen deferral/source comparison. No model or threshold selection."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The inherited entry rejects Rosetta/x86_64 before importing Torch.
from scripts.run_m3w_source_cost_deferral import load_config as load_training, build_data
from scripts.run_m3w_source_continuation import immutable_json
from scripts.run_m3w_source_start_probe import array_hash
import numpy as np
import torch
from src.world_model.m3w_source_cost_deferral import CostDeferral, predict_deferral
from src.world_model.m3w_source_cost_dynamics import SourceDynamics, forecast
from src.world_model.m3w_offline_visual_data import json_write
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_deferral_transfer import outputs, errors, seed_mean, grouped_summary
from src.evaluation.m3w_recording_diagnostic import error_summary, recording_resamples, paired_gain_interval


def load_config(path):
    reg = json.loads(path.read_text())
    if (reg['registration_path'] != str(path) or reg['role'] != 'fixed_explored_source_deferral_readout'
            or reg['seeds'] != [17,29,43] or reg['variants'] != ['expected_cost','cost_supervised']
            or reg['held_site'] != 'bookstore' or reg['step'] != 10000
            or reg['score_threshold'] != 0 or reg['bootstrap_resamples'] != 2000
            or reg['bootstrap_seed'] != 38113 or reg['training_updates'] != 0
            or reg['held_site_is_untouched'] or reg['model_selection'] or reg['new_deployment']
            or reg['main_primary_changed'] or not reg['bindings']):
        raise ValueError('Registered fixed source diagnostic required')
    for name, digest in reg['bindings'].items():
        if file_digest(ROOT/name) != digest:
            raise ValueError('Frozen dependency changed: '+name)
    return reg


def inventory(reg):
    previous = load_training(Path(reg['training_registration']))
    data, train, weights, scale, parents, controls = build_data(previous)
    report = json.loads((ROOT/reg['training_report']).read_text())
    analysis = json.loads((ROOT/reg['training_analysis']).read_text())
    assert report['completed_branches'] == 6 and report['additional_updates'] == 48000
    assert analysis['report_sha256'] == file_digest(ROOT/reg['training_report'])
    assert analysis['exact_milestone_replays'] == 24 and analysis['held_rows_scored'] == 0
    artifacts = dict(analysis['artifact_hashes'])
    for name, digest in artifacts.items():
        assert file_digest(ROOT/name) == digest
    held = np.flatnonzero(data.source_sites == reg['held_site'])+data.nmain
    loc, tr = held-data.nmain, train-data.nmain
    assert len(train) == 15430 and len(held) == 6944
    assert not np.intersect1d(train, held).size
    assert not set(data.source_tracks[tr]) & set(data.source_tracks[loc])
    assert not set(data.source_records[tr]) & set(data.source_records[loc])
    assert len(set(data.source_records[loc])) == 7 and len(set(data.source_tracks[loc])) == 181
    old = json.loads((ROOT/reg['dense_evaluation']).read_text())
    dense = [r for r in old['results'] if r['arm'] == 'mask_only' and r['schedule'] == 'cosine']
    assert {r['seed'] for r in dense} == set(reg['seeds'])
    frozen = []
    for t in report['trials']:
        assert t['variant'] in reg['variants'] and t['seed'] in reg['seeds']
        state = torch.load(ROOT/t['checkpoint_path'], map_location='cpu', weights_only=False)
        assert file_digest(ROOT/t['checkpoint_path']) == t['checkpoint_sha256']
        assert state['identity'] == t['identity'] and state['step'] == reg['step']
        np.testing.assert_array_equal(state['train_ids'], train)
        frozen.append(dict(variant=t['variant'], seed=t['seed'], trial=t['trial'],
            checkpoint_path=t['checkpoint_path'], checkpoint_sha256=t['checkpoint_sha256'],
            expected_identity=t['identity']))
    assert len({(f['variant'],f['seed']) for f in frozen}) == 6
    for r in dense:
        control = controls[r['seed']]
        assert r['checkpoint_sha256'] == control['checkpoint_sha256']
        assert file_digest(ROOT/r['prediction_path']) == r['prediction_sha256']
        artifacts[r['prediction_path']] = r['prediction_sha256']
        frozen.append(dict(variant='dense_control', seed=r['seed'], trial=r['trial'],
            checkpoint_path=r['checkpoint_path'], checkpoint_sha256=r['checkpoint_sha256'],
            expected_identity=control['identity'], cached_prediction_path=r['prediction_path'],
            cached_prediction_sha256=r['prediction_sha256']))
    for key in ('training_report','training_analysis','dense_evaluation'):
        artifacts[reg[key]] = file_digest(ROOT/reg[key])
    return data, train, held, scale, frozen, artifacts


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', required=True, type=Path)
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--replay', action='store_true')
    args = p.parse_args()
    if args.audit_only and args.replay:
        raise ValueError('Choose audit or replay')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg = load_config(args.registration)
    private, public = ROOT/reg['output'], ROOT/reg['reports']
    def beat(**value):
        event = dict(pid=os.getpid(), timestamp_unix=time.time(), **value)
        json_write(private/'heartbeat.json', event)
        print(json.dumps(event), flush=True)
    beat(state='verifying_frozen_assets')
    data, train, held, scale, frozen, before = inventory(reg)
    identity = dict(registration_sha256=file_digest(args.registration), data_identity=data.identity,
        source_assignment_sha256=data.assignment_hash, held_ids_sha256=array_hash(held),
        training_ids_sha256=array_hash(train), cost_scale=scale, torch=torch.__version__, numpy=np.__version__,
        torch_threads=4, interop_threads=1, num_workers=0)
    immutable_json(private/'identity.json', identity)
    immutable_json(public/'frozen_predictors.json', dict(identity=identity, predictors=frozen,
        independent_confirmation=False, threshold=0, model_selection=False, new_deployment=False))
    immutable_json(public/'input_checks.json', dict(identity=identity, training_rows=len(train), held_rows=len(held),
        source_recordings=7, scoped_agents=181, physical_sites=1, held_site_previously_explored=True,
        labels_in_inference=False, main_roles_scored=0, main_primary_changed=False, sealed_roles_opened=False,
        future_labels_for_evaluation_only=True, units='past_normalized_and_annotation_pixels_raw_frame_only'))
    if args.audit_only:
        beat(state='audit_complete_no_new_forecasts', frozen_neural=6, cached_dense=3)
        return
    records, cached, exact = [], {}, []
    for item in frozen:
        key, variant = item['trial'], item['variant']
        state = torch.load(ROOT/item['checkpoint_path'], map_location='cpu', weights_only=False)
        assert state['identity'] == item['expected_identity'] and state['step'] == 10000
        if variant == 'dense_control':
            model = SourceDynamics(); model.load_state_dict(state['model'])
            proposal = forecast(model, data.dynamics_inputs, held, 'mask_only')
            score = None
            with np.load(ROOT/item['cached_prediction_path'], allow_pickle=False) as old:
                np.testing.assert_array_equal(old['ids'], held)
                np.testing.assert_array_equal(old['prediction'], proposal)
            pp = ROOT/item['cached_prediction_path']
        else:
            model = CostDeferral(); model.load_state_dict(state['model'])
            proposal, score = predict_deferral(model, data.dynamics_inputs, held)
            pp = private/'predictions'/f'{key}.npz'
            receipt = pp.with_suffix('.json')
            if receipt.exists():
                saved = json.loads(receipt.read_text())
                assert saved['identity'] == identity and saved['checkpoint_sha256'] == item['checkpoint_sha256']
                assert saved['prediction_sha256'] == file_digest(pp)
                with np.load(pp, allow_pickle=False) as old:
                    np.testing.assert_array_equal(old['ids'], held)
                    np.testing.assert_array_equal(old['proposal'], proposal)
                    np.testing.assert_array_equal(old['score'], score)
            elif args.replay:
                raise ValueError('Replay requires saved predictions')
            else:
                pp.parent.mkdir(parents=True, exist_ok=True)
                temp = pp.with_suffix('.tmp.npz')
                np.savez(temp, ids=held, proposal=proposal, score=score)
                os.replace(temp, pp)
                json_write(receipt, dict(identity=identity, checkpoint_sha256=item['checkpoint_sha256'],
                                         prediction_sha256=file_digest(pp)))
        loc = held-data.nmain
        bound = data.radius[loc]*np.linalg.svd(data.rotation[loc], compute_uv=False)[:,0]
        assert np.isfinite(proposal).all() and not np.any(proposal[~data.support[loc]])
        assert np.all(np.linalg.norm(proposal.astype(float),axis=-1)<=bound[:,None]*(1+1e-5)+1e-8)
        emitted = outputs(proposal, score)
        cached[variant,item['seed']] = (emitted, score)
        records.append(dict(**item, prediction_path=str(pp.relative_to(ROOT)), prediction_sha256=file_digest(pp),
            result_source='cached_verified_with_fresh_exact_replay' if variant=='dense_control'
                          else 'fresh_run_inference_from_cached_verified_checkpoint'))
        exact.append(key)
        beat(state='prediction_replayed' if args.replay else 'frozen_prediction_complete', trial=key)
    assert len(exact) == 9
    assert before == {name:file_digest(ROOT/name) for name in before}
    if args.replay:
        immutable_json(public/'replay.json', dict(identity=identity, all_exact=True, predictors=exact,
            new_neural_replays=6, cached_dense_replays=3, new_training_updates=0, inherited_artifacts_unchanged=len(before)))
        beat(state='replay_complete', count=9, new_training_updates=0)
        return
    # All observed-input predictions are frozen before any label-based readout.
    loc, tr = held-data.nmain, train-data.nmain
    target, native = data.target[loc].astype(float), data.native_scale[loc]
    cv = np.linalg.norm(target, axis=-1).mean(1)
    train_cv = np.linalg.norm(data.target[tr].astype(float),axis=-1).mean(1)
    hard_cut = float(np.quantile(train_cv,.9)); hard = cv>=hard_cut
    group_labels, inverse, counts = recording_resamples(data.source_records[loc],reg['bootstrap_resamples'],reg['bootstrap_seed'])
    assert len(group_labels) == 7
    arrays, per_seed, summaries, group_rows, means = {}, [], [], [], {}
    for (variant,seed),(emitted,score) in cached.items():
        for mode, prediction in emitted.items():
            requested = score>0 if mode=='hard_action' else np.ones(len(held),bool)
            values = errors(prediction,target,requested)
            arrays[variant,mode,seed] = values
            row = error_summary(values['ade'],values['fde'],cv,native,values['changed'],hard)
            per_seed.append(dict(variant=variant,mode=mode,seed=seed,requested_rate=float(requested.mean()),**row))
    for variant in reg['variants']+['dense_control']:
        for mode in (['proposal'] if variant=='dense_control' else ['proposal','hard_action']):
            cells = [arrays[variant,mode,s] for s in reg['seeds']]
            mean = seed_mean(cells); means[variant,mode] = mean
            summary, groups = grouped_summary(mean,cv,native,hard,
                {'recording':data.source_records[loc],'scoped_agent':data.source_tracks[loc]},inverse,counts)
            gains = [100*(1-c['ade'].sum()/cv.sum()) for c in cells]
            summaries.append(dict(variant=variant,mode=mode,**summary,gain_seed_range=[min(gains),max(gains)]))
            group_rows.extend(dict(variant=variant,mode=mode,**g) for g in groups)
    contrasts = []
    for variant in reg['variants']:
        for mode in ['proposal','hard_action']:
            for ref in [('dense_control','proposal')]+([(variant,'proposal')] if mode=='hard_action' else []):
                for subset, mask in [('all',np.ones(len(cv),bool)),('hard',hard),('moving',cv>0)]:
                    contrasts.append(dict(candidate=f'{variant}:{mode}',reference=':'.join(ref),subset=subset,
                        **paired_gain_interval(means[ref]['ade'],means[variant,mode]['ade'],cv,inverse,counts,mask)))
    result = dict(result_source='fresh_run_fixed_inference_cached_verified_dense_controls',identity=identity,
        results=records,summaries=summaries,contrasts=contrasts, training_updates=0,held_rows=len(held),
        physical_sites=1,source_recordings=7,scoped_agents=181,training_hard_cut=hard_cut,
        bootstrap_resamples=2000,bootstrap_scope='conditional_on_seven_recordings_one_previously_explored_site',
        seed_aggregation='mean_errors_not_prediction_ensemble',independent_confirmation=False,
        inherited_artifact_hashes=before,main_primary_changed=False,sealed_roles_opened=False,
        new_deployment=False,stage5c_executed=False,smc_enabled=False)
    immutable_json(public/'evaluation.json',result)
    write_csv(public/'seed_metrics.csv',per_seed)
    write_csv(public/'recording_metrics.csv',group_rows)
    write_csv(public/'paired_contrasts.csv',contrasts)
    lines=['# Fixed Source-Site Deferral Readout','','## Material Passport','',
        'Fresh inference from all six frozen neural endpoints. Three dense controls are cached_verified and freshly replayed.',
        '6944 queries, seven recordings, 181 scoped agents, one previously explored physical site. No training or threshold selection.',
        'Seed errors are averaged, not forecasts. All intervals are conditional recording-block diagnostics, not independent scene confirmation.','',
        '| Variant | Output | Gain vs CV (%) | Conditional recording 95% CI | Hard gain (%) | Zero-target harm (pixels) | Actual intervention |',
        '| --- | --- | ---: | --- | ---: | ---: | ---: |']
    for s in summaries:
        lo,hi=s['gain_interval']['conditional_recording_ci95']
        lines.append(f"| {s['variant']} | {s['mode']} | {s['gain_percent']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {s['hard_gain_percent']:+.6f} | {s['easy_pixel_harm']:.8f} | {s['actual_changed_rate']:.3%} |")
    lines+=['','## Paired Aggregate Contrasts','','| Candidate | Reference | Difference (pp) | Conditional recording 95% CI |','| --- | --- | ---: | --- |']
    for c in contrasts:
        if c['subset']=='all':
            lo,hi=c['conditional_recording_ci95']
            lines.append(f"| {c['candidate']} | {c['reference']} | {c['point_percent']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines+=['','## Limits','',
        'Easy relative degradation is undefined at zero CV error; absolute harm is not a passed 2% gate.',
        'All-baseline output and improvement over a bad neural control do not establish positive gain over the baseline.',
        'Offline supplied annotations, stride12/+144 raw frames, annotation pixels/past-normalized coordinates only.',
        'No meters/seconds, human intention gold, true-3D, foundation, deployment or submission-readiness claim. Stage5C and SMC remain off.','']
    (public/'results.md').write_text('\n'.join(lines))
    plot_cache=private/'plot_runtime';plot_cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(plot_cache/'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt']='source-deferral-transfer-v1'
    fig,axes=plt.subplots(1,2,figsize=(11,4.7))
    names=[f"{s['variant']}\n{s['mode']}" for s in summaries]
    for i,s in enumerate(summaries):
        lo,hi=s['gain_interval']['conditional_recording_ci95']
        axes[0].plot(i,s['gain_percent'],'o',color='#357289'); axes[0].vlines(i,lo,hi,color='#357289')
        axes[1].bar(i,s['easy_pixel_harm'],color='#a45351',width=.6)
    for ax in axes:
        ax.set_xticks(range(len(names)),names,rotation=25,ha='right',fontsize=8)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
    axes[0].axhline(0,color='black',linewidth=.7,linestyle=':')
    axes[0].set_ylabel('Held-source gain vs stationary CV (%)')
    axes[1].set_ylabel('Zero-target harm (annotation pixels)')
    fig.suptitle('Frozen endpoints on one previously explored source site')
    fig.text(.5,.01,'2000 paired recording resamples; not independent scene confirmation. No model selection.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.96))
    fig.savefig(public/'comparison.svg',metadata={'Date':None});fig.savefig(plot_cache/'comparison.png',dpi=150);plt.close(fig)
    path=public/'comparison.svg';path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines())+'\n')
    beat(state='fixed_evaluation_complete',new_training_updates=0,new_predictors=6,cached_dense=3)


if __name__=='__main__':
    main()
