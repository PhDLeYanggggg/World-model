"""Verify complete training-only microfits and retain every seed/cohort/decoder."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_m3w_source_microfit import load_config, build_data
import numpy as np
import torch
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    reg=load_config(args.registration);out=ROOT/reg['output'];reports=ROOT/reg['reports'];rp=reports/'report.json'
    report=json.loads(rp.read_text());assert report['completed_models']==12 and report['optimizer_updates']==24000
    replay=json.loads((reports/'replay.json').read_text());assert replay['identity']==report['identity']
    assert replay['all_exact'] and len(replay['exact_trials'])==12 and replay['new_updates']==0
    data,scale,payloads,audit=build_data(reg);assert audit==report['identity']['cohorts']
    lookup={(t['cohort'],t['decoder'],t['seed']):t for t in report['trials']}
    rows=[];pairs=[]
    for cohort in reg['cohorts']:
        ids,features,frame,target=payloads[cohort]
        for seed in reg['seeds']:
            states=[]
            for decoder in reg['decoders']:
                t=lookup[cohort,decoder,seed]
                for name in ('checkpoint','prediction'):
                    assert file_digest(ROOT/t[name+'_path'])==t[name+'_sha256']
                cp=torch.load(ROOT/t['checkpoint_path'],map_location='cpu',weights_only=False)
                assert cp['identity']==t['identity'] and cp['config']==reg['training']
                assert cp['step']==2000 and cp['examples_per_step']==len(ids) and cp['scale']==scale
                assert all(torch.isfinite(v).all() for v in cp['model'].values())
                assert 0<=cp['clipped_updates']<=cp['step'] and np.isfinite(cp['max_gradient'])
                with np.load(ROOT/t['prediction_path'],allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['ids'],ids);prediction=saved['prediction']
                assert np.isfinite(prediction).all() and prediction.shape==tuple(target.shape)
                bound=frame[0].numpy()*np.linalg.svd(frame[1].numpy(),compute_uv=False)[:,0]
                assert np.all(np.linalg.norm(prediction,axis=-1)<=bound[:,None]+1e-5*bound[:,None])
                m=t['fit']['final'];ade=np.linalg.norm(prediction.astype(float)-target.numpy(),axis=-1).mean()
                np.testing.assert_allclose(ade,m['ade'],rtol=1e-5,atol=1e-3)
                assert t['fit']['trace'][0]['gain_percent']==0
                rows.append(dict(cohort=cohort,decoder=decoder,seed=seed,rows=len(ids),
                    **m,clipped_update_fraction=cp['clipped_updates']/cp['step'],
                    max_preclip_gradient=cp['max_gradient'],fit_seconds=t['fit']['fit_seconds'],
                    updates=cp['step'],passes_per_row=cp['step']))
                states.append(cp)
            assert torch.equal(states[0]['torch_rng'],states[1]['torch_rng'])
            assert states[0]['examples_per_step']==states[1]['examples_per_step']
            pairs.append(dict(cohort=cohort,seed=seed,same_input_target_identity=True,same_fullbatch_exposure=True,same_rng_state=True))
    paths=sorted([*out.glob('checkpoints/*.pt'),*out.glob('predictions/*.npz'),*out.glob('trials/*.json'),out/'identity.json'])
    assert len(paths)==37
    hashes={str(p.relative_to(ROOT)):file_digest(p) for p in paths};report_hash=file_digest(rp)
    proc=subprocess.run([sys.executable,'scripts/run_m3w_source_microfit.py','--registration',str(args.registration)],
        cwd=ROOT,capture_output=True,text=True,check=True)
    event=[json.loads(x) for x in proc.stdout.splitlines() if x.startswith('{')][-1]
    assert event['state']=='completed_resume_verified' and event['new_updates']==event['new_fits']==0
    assert hashes=={str(p.relative_to(ROOT)):file_digest(p) for p in paths} and file_digest(rp)==report_hash
    summary=[]
    for cohort in reg['cohorts']:
        for decoder in reg['decoders']:
            cells=[r for r in rows if r['cohort']==cohort and r['decoder']==decoder]
            gains=[r['gain_percent'] for r in cells]
            easy=[r['easy_absolute_harm'] for r in cells]
            summary.append(dict(cohort=cohort,decoder=decoder,
                mean_ade=float(np.mean([r['ade'] for r in cells])),cv_ade=cells[0]['cv_ade'],
                gain_percent_mean=float(np.mean(gains)),gain_percent_seed_range=[min(gains),max(gains)],
                easy_harm_mean=float(np.mean(easy)) if easy[0] is not None else None,
                clipped_update_fraction=float(np.mean([r['clipped_update_fraction'] for r in cells])),
                positive_training_fits=sum(x>0 for x in gains),rows=cells[0]['rows']))
    evidence=dict(result_source='fresh_run_verification_and_training_only_analysis',report_sha256=report_hash,
        exact_replays=12,paired_checks=pairs,unchanged_artifacts=37,completed_resume=event,
        summary=summary,artifacts=hashes,held_rows_scored=0,new_deployment=False,
        generalization_established=False,decoder_also_changes_inductive_bias=True,
        selection_uses_training_labels=True,independent_confirmation=False)
    json_write(reports/'analysis.json',evidence)
    with (reports/'fit_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    traces=[dict(trial=t['trial'],**r) for t in report['trials'] for r in t['fit']['trace']]
    with (reports/'loss_trace.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(traces[0]),lineterminator='\n');writer.writeheader();writer.writerows(traces)
    lines=['# Training-Only Microfit Results','','## Material Passport','',
        'Fresh Torch optimization diagnostic on selected training rows. No held-site or main benchmark result.',
        'Twelve models,24000 updates; full-batch2000 passes per row. This differs from the full-source exposure budget.',
        'Three-seed ranges are descriptive optimizer variation, not confidence intervals for generalization.','',
        '| Cohort | Decoder | Rows | Final ADE | CV ADE | Mean gain (%) [seed range] | Easy absolute harm | Clipped updates |',
        '| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |']
    for s in summary:
        lo,hi=s['gain_percent_seed_range'];easy=s['easy_harm_mean'];easy='n/a' if easy is None else f'{easy:.6f}'
        lines.append(f"| {s['cohort']} | {s['decoder']} | {s['rows']} | {s['mean_ade']:.6f} | {s['cv_ade']:.6f} | {s['gain_percent_mean']:+.4f} [{lo:+.4f}, {hi:+.4f}] | {easy} | {s['clipped_update_fraction']:.2%} |")
    lines+=['','All12 forecasts replay exactly; six decoder pairs have matched full-batch exposure/RNG state.',
        'Completed resume preserves37 artifacts and the report with zero new updates. Targets are only loss labels.',
        'Feasible training-label cohort selection is not a deployment filter. Easy relative degradation is undefined.',
        'No independent confirmation, held-source score, main protocol change, Stage5C orSMC.','']
    (reports/'results.md').write_text('\n'.join(lines))
    cache=out/'plot_runtime';cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache/'matplotlib'));os.environ.setdefault('XDG_CACHE_HOME',str(cache/'xdg'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt']='m3w-source-microfit-v1'
    fig,axes=plt.subplots(1,2,figsize=(10,4.6))
    for ax,cohort in zip(axes,reg['cohorts']):
        for decoder,color in zip(reg['decoders'],['#357289','#a45351']):
            ts=[lookup[cohort,decoder,s]['fit']['trace'] for s in reg['seeds']]
            steps=[v['step'] for v in ts[0]];ys=np.array([[v['ade']/v['cv_ade'] for v in t] for t in ts])
            ax.plot(steps,ys.mean(0),label=decoder,color=color)
            ax.fill_between(steps,ys.min(0),ys.max(0),color=color,alpha=.15)
        ax.axhline(1,color='black',linestyle=':',linewidth=.8);ax.set_title(cohort)
        ax.set_xlabel('Optimizer updates / full passes');ax.set_ylabel('Training ADE / stationary CV ADE')
        ax.legend(fontsize=8);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Selected source-training microcohorts: mean and range of three seeds')
    fig.text(.5,.01,'Memorization/conditioning diagnostic only. No held-source evaluation or deployment claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.95));fig.savefig(reports/'training_trace.svg',metadata={'Date':None})
    fig.savefig(cache/'training_trace.png',dpi=140);plt.close(fig)
    svg=reports/'training_trace.svg';svg.write_text('\n'.join(x.rstrip() for x in svg.read_text().splitlines())+'\n')
    print(json.dumps({k:v for k,v in evidence.items() if k!='artifacts'},indent=2))


if __name__=='__main__':main()
