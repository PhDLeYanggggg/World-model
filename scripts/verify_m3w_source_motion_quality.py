"""Independent artifact, role, score and immutable-resume checks for motion probes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_m3w_source_motion_quality import load_config, load_parent, context, probability_metrics
from scripts.run_m3w_source_continuation import immutable_json
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_source_motion_quality import trajectory_quality
import joblib
import numpy as np
import torch


def preserve_verification(path, report):
    if path.exists():
        previous = json.loads(path.read_text())
        # Resume PID/time changes on recheck; scientific receipts must not.
        stable = lambda value: {k: v for k, v in value.items() if k != 'completed_resume'}
        if stable(previous) != stable(report):
            raise ValueError('Completed verification evidence changed')
    else:
        immutable_json(path, report)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    reg=load_config(args.registration);public,private=ROOT/reg['reports'],ROOT/reg['output']
    prep=json.loads((public/'preparation.json').read_text());probes=json.loads((public/'probes.json').read_text())
    replay=json.loads((public/'replay.json').read_text());analysis=json.loads((public/'analysis.json').read_text())
    assert len(set(replay['replayed_models']))==48 and replay['all_bitwise_exact']
    assert file_digest(private/'rows.npz')==prep['rows_sha256']
    with np.load(private/'rows.npz',allow_pickle=False) as values: a={k:values[k].copy() for k in values.files}
    _,_,data=context(load_parent(Path(reg['parent_registration'])))
    np.testing.assert_array_equal(a['ids'],np.flatnonzero(data.source_sites!='bookstore')+data.nmain)
    np.testing.assert_array_equal(a['sites'],data.source_sites[a['loc']])
    quality=trajectory_quality(a['future_native'],a['past_box_scale'])
    before={}; keys=set(); score_checks=0; target_checks=0
    for trial in probes['trials']:
        identity=trial['identity'];site=identity['site'];label=identity['label'];arm=identity['arm']
        key=(site,label,arm,identity['seed']);assert key not in keys;keys.add(key)
        train,_,held,outer,_=data.configure(site)
        assert not np.intersect1d(train,held).size and not np.intersect1d(train,outer).size
        tr=np.flatnonzero(a['sites']!=site);ho=np.flatnonzero(a['sites']==site)
        cp,pp=ROOT/trial['model_path'],ROOT/trial['prediction_path']
        assert file_digest(cp)==trial['model_sha256'] and file_digest(pp)==trial['prediction_sha256']
        saved=joblib.load(cp);assert saved['identity']==identity
        np.testing.assert_array_equal(saved['train_ids'],train)
        assert saved['model'].n_features_in_==480+(38 if arm=='geometry_past_box' else 0)
        assert saved['model'].random_state==identity['seed']
        for name,value in reg['forest'].items(): assert saved['model'].get_params()[name]==value
        y=quality[label].astype(int);reference=float(y[tr].mean())
        with np.load(pp,allow_pickle=False) as pred:
            np.testing.assert_array_equal(pred['ids'],held);np.testing.assert_array_equal(pred['target'],y[ho])
            np.testing.assert_array_equal(pred['reference'],np.full(len(ho),reference))
            assert probability_metrics(y[ho],pred['probability'],reference)==trial['metrics']
        original=data.z[a['ids']].copy();target=data.target.copy()
        try:
            data.target[:]=np.nan
            np.testing.assert_array_equal(original,data.z[a['ids']])
        finally: data.target[:]=target
        target_checks+=1;score_checks+=1
        before[str(cp.relative_to(ROOT))]=file_digest(cp);before[str(pp.relative_to(ROOT))]=file_digest(pp)
    assert len(keys)==48
    for field in ('max_pixel_displacement','displacement_over_past_box'):
        assert sum(v['rows'] for v in analysis['strata'][field].values())==15430
    for path in [*private.joinpath('trials').glob('*.json'),private/'rows.npz',public/'preparation.json',
                 public/'probes.json',public/'replay.json',public/'analysis.json']:
        before[str(path.relative_to(ROOT))]=file_digest(path)
    run=subprocess.run([sys.executable,'scripts/run_m3w_source_motion_quality.py','--registration',str(args.registration),
        '--phase','probe'],cwd=ROOT,capture_output=True,text=True,check=True)
    event=[json.loads(line) for line in run.stdout.splitlines() if line.startswith('{')][-1]
    assert event['state']=='probes_complete' and event['new_fits']==0
    assert before=={name:file_digest(ROOT/name) for name in before}
    report=dict(result_source='fresh_run_role_score_replay_and_immutable_resume_verification',
        registration_sha256=file_digest(args.registration),exact_replayed_forests=48,
        score_and_train_prevalence_recomputations=score_checks,loaded_label_isolation_checks=target_checks,
        raw_future_box_poison_checks=prep['raw_future_box_poison_checks'],all_magnitude_bins_retain_all_rows=True,
        immutable_artifacts=len(before),artifact_hashes=before,completed_resume=event,new_fits_on_resume=0,
        outer_rows_scored=0,main_rows_scored=0,new_deployment=False,
        labels_are_not_human_gold=True,sensor_asof_certified=False)
    preserve_verification(public/'verification.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='artifact_hashes'},indent=2))


if __name__=='__main__':main()
