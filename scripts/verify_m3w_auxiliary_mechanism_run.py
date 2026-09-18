"""Verify completed phase/sample identities without fitting or selecting models."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch

from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_auxiliary_mechanism import load_mechanism_registration, NEW_ARMS
from src.world_model.m3w_offline_visual_data import json_write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    reg,parent=load_mechanism_registration(ROOT,args.registration)
    report_path=ROOT/reg['reports']/'report.json'
    report=json.loads(report_path.read_text())
    if not report['complete'] or len(report['trials'])!=108:
        raise ValueError('All 54 new and 54 cached controls required')
    look={(t['arm'],t['modality'],t['seed'],t['fold']):t for t in report['trials']}
    if len(look)!=108:
        raise ValueError('Repeated trial key')
    checks=[]; steps=0
    for arm in NEW_ARMS:
        for modality in reg['modalities']:
            for seed in reg['seeds']:
                for fold in range(3):
                    t=look[arm,modality,seed,fold]
                    ref=look['sdd_aux',modality,seed,fold]
                    for item in (t,ref):
                        for kind in ('checkpoint','prediction'):
                            if file_digest(ROOT/item[kind+'_path'])!=item[kind+'_sha256']:
                                raise ValueError('Changed completed artifact')
                    cp=torch.load(ROOT/t['checkpoint_path'],map_location='cpu',weights_only=False)
                    old=torch.load(ROOT/ref['checkpoint_path'],map_location='cpu',weights_only=False)
                    expected=4000 if arm=='main4k' else 6000
                    assert cp['identity']==t['identity'] and cp['step']==expected
                    assert cp['identity']['registration_sha256']==file_digest(args.registration)
                    assert all(torch.isfinite(v).all() for v in cp['model'].values())
                    assert all(np.isfinite(x['loss']) and np.isfinite(x['gradient_norm']) for x in cp['losses'])
                    np.testing.assert_array_equal(cp['draw_counts']['main'],old['draw_counts']['main'])
                    assert torch.equal(cp['sampler_rng'],old['sampler_rng'])
                    assert cp['draw_counts']['main'].sum()==256000
                    if arm=='sdd_permuted':
                        np.testing.assert_array_equal(cp['draw_counts']['auxiliary'],old['draw_counts']['auxiliary'])
                        assert cp['draw_counts']['auxiliary'].sum()==128000
                    else:
                        assert not cp['draw_counts']['auxiliary'].any()
                    steps+=cp['step']
                    checks.append(dict(trial=t['trial'],checkpoint_steps=cp['step'],
                        same_main_draw_counts=True,same_final_main_sampler_state=True,
                        same_source_draw_counts=arm=='sdd_permuted',
                        no_source_draws=arm=='main4k',finite_parameters_and_logged_gradients=True))
    assert steps==270000
    result=dict(result_source='fresh_run_checkpoint_sample_identity_verification',
        report_sha256=file_digest(report_path),registration_sha256=file_digest(args.registration),
        verified_new_fits=54,verified_new_optimizer_steps=steps,new_training_updates=0,
        checks=checks,matched_sample_counts_and_final_sampler=True,
        exact_full_stream_order_covered_by_phase_sampler_unit_tests=True,
        prediction_replay_is_separate=True,independent_confirmation=False)
    json_write(ROOT/reg['reports']/'stream_verification.json',result)
    print(json.dumps(dict(verified_new_fits=54,verified_new_steps=steps,new_training_updates=0)))


if __name__=='__main__':
    main()
