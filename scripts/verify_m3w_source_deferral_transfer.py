"""Verify frozen source readout, rowwise statistics and repeat-run immutability."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_m3w_source_deferral_transfer import load_config, inventory
import numpy as np
import torch
from src.evaluation.m3w_deferral_transfer import outputs, errors, seed_mean, grouped_summary
from src.evaluation.m3w_recording_diagnostic import error_summary, recording_resamples
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    reg=load_config(args.registration)
    private,public=ROOT/reg['output'],ROOT/reg['reports']
    report=json.loads((public/'evaluation.json').read_text())
    replay=json.loads((public/'replay.json').read_text())
    assert replay['identity']==report['identity'] and replay['all_exact']
    assert len(set(replay['predictors']))==9 and replay['new_training_updates']==0
    data,train,held,scale,frozen,before=inventory(reg)
    assert before==report['inherited_artifact_hashes']
    loc,tr=held-data.nmain,train-data.nmain
    target,native=data.target[loc].astype(float),data.native_scale[loc]
    cv=np.linalg.norm(target,axis=-1).mean(1)
    cutoff=float(np.quantile(np.linalg.norm(data.target[tr].astype(float),axis=-1).mean(1),.9))
    assert cutoff==report['training_hard_cut']
    hard=cv>=cutoff
    _,inverse,counts=recording_resamples(data.source_records[loc],2000,38113)
    table=list(csv.DictReader((public/'seed_metrics.csv').open()))
    arrays={}
    for result,item in zip(report['results'],frozen):
        assert all(result[k]==v for k,v in item.items())
        pp=ROOT/result['prediction_path']
        assert file_digest(pp)==result['prediction_sha256']
        with np.load(pp,allow_pickle=False) as a:
            np.testing.assert_array_equal(a['ids'],held)
            proposal=a['prediction'].astype(float) if item['variant']=='dense_control' else a['proposal'].astype(float)
            score=None if item['variant']=='dense_control' else a['score'].copy()
        before[str(pp.relative_to(ROOT))]=file_digest(pp)
        if score is not None:
            receipt=pp.with_suffix('.json')
            saved=json.loads(receipt.read_text())
            assert saved['prediction_sha256']==file_digest(pp) and saved['identity']==report['identity']
            before[str(receipt.relative_to(ROOT))]=file_digest(receipt)
        for mode,prediction in outputs(proposal,score).items():
            requested=score>0 if mode=='hard_action' else np.ones(len(cv),bool)
            values=errors(prediction,target,requested)
            arrays[item['variant'],mode,item['seed']]=values
            actual=error_summary(values['ade'],values['fde'],cv,native,values['changed'],hard)
            row=next(r for r in table if r['variant']==item['variant'] and r['mode']==mode and int(r['seed'])==item['seed'])
            for field in ('ade','fde','gain_percent','native_pixel_ade','easy_pixel_harm','actual_changed_rate','tail_ade95','tail_ade99'):
                np.testing.assert_allclose(float(row[field]),actual[field],rtol=1e-12,atol=1e-12)
            if mode=='hard_action':
                assert not np.any(prediction[score<=0])
    assert len(arrays)==15
    for summary in report['summaries']:
        mean=seed_mean([arrays[summary['variant'],summary['mode'],seed] for seed in reg['seeds']])
        actual,_=grouped_summary(mean,cv,native,hard,
            {'recording':data.source_records[loc],'scoped_agent':data.source_tracks[loc]},inverse,counts)
        for field,value in actual.items():
            assert summary[field]==value
    paths=[private/'identity.json',public/'frozen_predictors.json',public/'input_checks.json',
        public/'evaluation.json',public/'replay.json',public/'seed_metrics.csv',public/'recording_metrics.csv',
        public/'paired_contrasts.csv',public/'results.md',public/'comparison.svg']
    before.update({str(path.relative_to(ROOT)):file_digest(path) for path in paths})
    process=subprocess.run([sys.executable,'scripts/run_m3w_source_deferral_transfer.py','--registration',str(args.registration)],
                           cwd=ROOT,capture_output=True,text=True,check=True)
    event=[json.loads(line) for line in process.stdout.splitlines() if line.startswith('{')][-1]
    assert event['state']=='fixed_evaluation_complete' and event['new_training_updates']==0
    assert before=={name:file_digest(ROOT/name) for name in before}
    evidence=dict(result_source='fresh_run_exact_replay_and_readonly_verification',
        identity=report['identity'],new_neural_prediction_score_replays=6,cached_dense_exact_replays=3,
        seed_output_cells_recomputed=15,three_seed_summaries_recomputed=5,
        unchanged_artifacts=len(before),artifact_hashes=before,repeated_evaluation=event,
        new_training_updates=0,main_roles_scored=0,independent_confirmation=False,new_deployment=False)
    json_write(public/'verification.json',evidence)
    print(json.dumps({k:v for k,v in evidence.items() if k not in ('artifact_hashes','identity')},indent=2))


if __name__=='__main__':
    main()
