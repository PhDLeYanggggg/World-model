"""Fixed conditioning/decoder/loss mechanism comparison, using fit roles only."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '4'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.run_m3w_auxiliary_mechanism import evaluate
from scripts.run_m3w_objective_alignment import paired_interval
from scripts.run_m3w_sdd_auxiliary import Corpus
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_objective_alignment import supported_standardization
from src.world_model.m3w_observed_unit_frame_v2 import observed_unit_frame
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_offline_visual_forecast import OfflineVisualForecast, forecast_metrics
from src.world_model.m3w_sdd_auxiliary import load_registration
from src.world_model.m3w_unit_frame_training import ARMS, fit_frame, frame_prediction


def internal_arrays(x):
    chunks = [observed_unit_frame(x[i:i+4096]) for i in range(0,len(x),4096)]
    return tuple(np.concatenate([c[k] for c in chunks]) for k in range(4))


class FrameCorpus(Corpus):
    def __init__(self, parent):
        super().__init__(parent)
        self.main_frame = internal_arrays(self.main['geometry'])
        self.source_frame = internal_arrays(self.aux['geometry'])
        self.arm = None

    def set_fold(self, fold):
        super().set_fold(fold)
        self.x, norm = supported_standardization(self.main_frame[0][self.train],self.main_frame[0])
        z = (self.source_frame[0]-norm['mean'])/norm['std']
        self.aux_clip_fraction = float((np.abs(z[:,~norm['constant']])>10).mean())
        self.ax = np.clip(z,-10,10).astype(np.float32)
        self.ax[:,norm['constant']] = 0
        self.normalizer = norm

    def batch(self, ids, modality='geometry', *, auxiliary=False, training=False):
        if modality != 'geometry':
            raise ValueError('This mechanism experiment is geometry-only')
        value = super().batch(ids,modality,auxiliary=auxiliary,training=training)
        frame = self.source_frame if auxiliary else self.main_frame
        for name, column in (('radius',1),('rotation',2),('spatial_support',3)):
            value[name] = torch.from_numpy(frame[column][ids].copy())
        return value

    def infer(self, model, ids, modality='geometry'):
        if self.arm not in ARMS:
            raise ValueError('Explicit prediction arm required')
        model.eval()
        with torch.no_grad():
            return np.concatenate([frame_prediction(model,self.batch(ids[i:i+128],modality),self.arm).numpy()
                                   for i in range(0,len(ids),128)])


def registration(path):
    reg = json.loads(path.read_text())
    if (str(path) != reg['registration_path'] or reg['arms'] != list(ARMS)
            or reg['role'] != 'fit_only_exploratory' or reg['modalities'] != ['geometry']):
        raise ValueError('Fixed geometry-only registration required')
    for name, digest in reg['bindings'].items():
        if file_digest(ROOT/name) != digest:
            raise ValueError('Registered dependency changed: '+name)
    parent = load_registration(ROOT,ROOT/reg['parent_registration'])
    if reg['training'] != parent['training'] or reg['seeds'] != parent['seeds']:
        raise ValueError('Matched budget and seeds required')
    return reg,parent


def controls(reg,data):
    prior = json.loads((ROOT/reg['parent_report']).read_text())
    if not prior['complete']:
        raise ValueError('Completed parent required')
    for k,v in data.identity.items():
        if prior['identity'][k] != v:
            raise ValueError('Parent population changed')
    rows = [t for t in prior['trials'] if t['schedule']=='sdd_aux' and t['modality']=='geometry']
    if len(rows) != 9:
        raise ValueError('Nine paired parent controls required')
    verified = []
    for t in rows:
        for kind in ('checkpoint','prediction'):
            if file_digest(ROOT/t[kind+'_path']) != t[kind+'_sha256']:
                raise ValueError('Parent artifact changed')
        ids = np.flatnonzero(data.main['folds']==t['fold'])
        with np.load(ROOT/t['prediction_path'],allow_pickle=False) as saved:
            np.testing.assert_array_equal(ids,saved['held_indices'])
            p = saved['prediction'].copy()
        no_anchor = ~data.main_frame[3][ids]
        p[no_anchor] = data.main['baselines'][ids[no_anchor],data.cv]
        guard = forecast_metrics(p,data.main['targets'][ids],data.main['baselines'][ids,data.cv],
                                 data.main['scale'][ids],data.easy_threshold)
        verified.append(dict(t,arm='legacy_sdd_aux',result_source='cached_verified_parent',
            no_anchor_guard_only_counterfactual=guard,no_anchor_rows=int(no_anchor.sum())))
    return verified


def summaries(trials,reg):
    lookup = {(t['arm'],t['seed'],t['fold']):t for t in trials}
    def matrix(arm,field='primary_ADE'):
        return np.array([[lookup[arm,s,f]['vs_CV'][field] for f in range(3)] for s in reg['seeds']])
    out = {}
    for arm in ('legacy_sdd_aux',)+ARMS:
        ts = [t for t in trials if t['arm']==arm]
        out[arm] = dict(vs_CV=paired_interval(matrix(arm),matrix(arm,'reference_ADE'),2000),
            vs_legacy=paired_interval(matrix(arm),matrix('legacy_sdd_aux'),2000),
            vs_inputs_only=paired_interval(matrix(arm),matrix('unit_inputs_only'),2000),
            positive_fits=sum(t['vs_CV']['improvement_percent']>0 for t in ts),
            safe_positive_fits=sum(t['vs_CV']['improvement_percent']>0 and
                t['vs_CV']['easy_degradation_percent'] is not None and t['vs_CV']['easy_degradation_percent']<=2 for t in ts),
            easy_degradation_percent=[t['vs_CV']['easy_degradation_percent'] for t in ts],
            easy_absolute_harm=[t['vs_CV']['easy_absolute_harm'] for t in ts],
            training_gains=[t['train_equal_scene_gain_percent'] for t in ts],
            fit_seconds=sum(t['fit']['fit_seconds'] for t in ts))
    return out


def report_files(result,path):
    json_write(path/'report.json',result)
    lines = ['# Observed-Unit Conditioning: Fixed Fit-Only Comparison','',
        '27 fresh geometry-only Torch fits; nine cached_verified geometry controls. No held-score selection.',
        '8 observed / 12 predicted native annotation steps. Primary evaluation remains past-normalized ADE, equal physical site.',
        'SDD auxiliary supervision remains 2,000 updates, followed by 4,000 main updates. No sealed roles opened.','',
        '| Arm | Gain vs CV (%) | Gain vs legacy (%) | Safe positive fits |',
        '| --- | ---: | ---: | ---: |']
    for key,s in result['summary'].items():
        lines.append(f"| {key} | {s['vs_CV']['gain_percent']:.6f} | {s['vs_legacy']['gain_percent']:.6f} | {s['safe_positive_fits']}/9 |")
    lines += ['','All exposed-site results, including easy harm, are retained. Three-site bootstrap is descriptive, not independent confirmation.',
        'Input repair is not a contribution claim; internal training loss is not a change to the primary evaluation metric.',
        'No new multimodal performance claim, deployment, metric/seconds, Stage5C or SMC.','']
    (path/'conclusions.md').write_text('\n'.join(lines))
    stream=io.StringIO(); fields=['trial','source','arm','seed','fold','gain_vs_CV','ADE','FDE','easy_degradation','easy_absolute_harm','train_gain','fit_seconds']
    writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n'); writer.writeheader()
    for t in result['trials']:
        m=t['vs_CV']
        writer.writerow(dict(trial=t['trial'],source=t['result_source'],arm=t['arm'],seed=t['seed'],fold=t['fold'],
            gain_vs_CV=m['improvement_percent'],ADE=m['primary_ADE'],FDE=m['FDE'],easy_degradation=m['easy_degradation_percent'],
            easy_absolute_harm=m['easy_absolute_harm'],train_gain=t['train_equal_scene_gain_percent'],fit_seconds=t['fit']['fit_seconds']))
    (path/'fit_metrics.csv').write_text(stream.getvalue())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--trial'); parser.add_argument('--stop-at',type=int); parser.add_argument('--replay',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg,parent=registration(args.registration)
    keys={f'{a}_seed{s}_fold{f}' for a in ARMS for s in reg['seeds'] for f in range(3)}
    if args.trial and args.trial not in keys:
        raise ValueError('Unregistered trial')
    if args.stop_at is not None and (not args.trial or args.replay):
        raise ValueError('Pilot requires one training trial and no replay')
    out,reports=ROOT/reg['output'],ROOT/reg['reports']
    def heartbeat(**v):
        event=dict(pid=os.getpid(),timestamp_unix=time.time(),**v)
        json_write(out/'heartbeat.json',event); print(json.dumps(event),flush=True)
    heartbeat(state='verifying_inputs')
    data=FrameCorpus(parent); old=controls(reg,data)
    identity=dict(registration_sha256=file_digest(args.registration),torch=torch.__version__,numpy=np.__version__,**data.identity)
    ip=out/'identity.json'
    if ip.exists() and json.loads(ip.read_text()) != identity:
        raise ValueError('Runtime or population identity changed')
    json_write(ip,identity)
    checks=dict(identity=identity,main_rows=len(data.main['geometry']),source_rows=data.auxiliary_rows,
        main_no_anchor_rows=int((~data.main_frame[3]).sum()),source_no_anchor_rows=int((~data.source_frame[3]).sum()),
        source_valid_label_rows=int(data.aux['valid'].any(1).sum()),
        source_internal_loss_rows=int((data.aux['valid'].any(1)&data.source_frame[3]).sum()),
        cached_verified_fits=9,main_primary_changed=False,sealed_roles_opened=False,source_val_test_opened=False)
    json_write(reports/'input_checks.json',checks)
    trials,replays,updates=[],[],0
    for fold in range(3):
        data.set_fold(fold)
        for seed in reg['seeds']:
            for arm in ARMS:
                key=f'{arm}_seed{seed}_fold{fold}'
                if args.trial and args.trial != key:
                    continue
                data.arm=arm
                cp,pp,rp=[out/d/(key+ext) for d,ext in (('checkpoints','.pt'),('predictions','.npz'),('trials','.json'))]
                ti=dict(identity,arm=arm,seed=seed,fold=fold)
                saved=json.loads(rp.read_text()) if rp.exists() else None
                if saved:
                    if saved['identity'] != ti or file_digest(cp)!=saved['checkpoint_sha256'] or file_digest(pp)!=saved['prediction_sha256']:
                        raise ValueError('Completed artifact changed')
                    if not args.replay:
                        trials.append(saved); continue
                torch.manual_seed(seed); model=OfflineVisualForecast(476)
                if args.replay:
                    if not saved:
                        raise ValueError('Replay requires completed trial')
                    state=torch.load(cp,map_location='cpu',weights_only=False)
                    if state['identity'] != ti or state['step'] != 6000:
                        raise ValueError('Replay checkpoint mismatch')
                    model.load_state_dict(state['model'])
                else:
                    fit=fit_frame(model,lambda ids:data.batch(data.train[ids],training=True),
                        lambda ids:data.batch(ids,auxiliary=True,training=True),[len(data.train),data.auxiliary_rows],
                        arm=arm,config=reg['training'],seed=seed,identity=ti,checkpoint=cp,
                        heartbeat=lambda **v:heartbeat(trial=key,**v),stop_at=args.stop_at)
                    updates+=fit['new_updates_this_invocation']
                    if not fit['complete']:
                        heartbeat(state='pilot_saved_no_held_evaluation',trial=key,**fit); return
                prediction=data.infer(model,data.held)
                if args.replay:
                    with np.load(pp,allow_pickle=False) as saved_prediction:
                        np.testing.assert_array_equal(prediction,saved_prediction['prediction'])
                        np.testing.assert_array_equal(data.held,saved_prediction['held_indices'])
                    replays.append(dict(trial=key,prediction_exact=True)); continue
                pp.parent.mkdir(parents=True,exist_ok=True)
                tmp=pp.with_suffix('.tmp.npz'); np.savez(tmp,prediction=prediction,held_indices=data.held); os.replace(tmp,pp)
                result=dict(identity=ti,trial=key,arm=arm,modality='geometry',seed=seed,fold=fold,
                    result_source='fresh_run_torch',fit=fit,source_clipped_fraction=data.aux_clip_fraction,
                    **evaluate(model,data,'geometry',prediction),
                    checkpoint_path=str(cp.relative_to(ROOT)),checkpoint_sha256=file_digest(cp),
                    prediction_path=str(pp.relative_to(ROOT)),prediction_sha256=file_digest(pp))
                json_write(rp,result); trials.append(result)
                heartbeat(state='held_fit_evaluated',trial=key,gain=result['vs_CV']['improvement_percent'],
                    easy=result['vs_CV']['easy_degradation_percent'])
    if args.replay:
        json_write(reports/'replay.json',dict(identity=identity,trials=replays,new_updates=0,all_exact=True))
        heartbeat(state='replay_complete',trials=len(replays)); return
    if args.trial:
        return
    if len(trials)!=27:
        raise ValueError('Incomplete fixed matrix')
    report_path=reports/'report.json'
    if report_path.exists() and updates==0:
        prior=json.loads(report_path.read_text())
        if prior['identity']!=identity or prior['trials']!=old+trials:
            raise ValueError('Completed report changed')
        heartbeat(state='completed_resume_verified',new_optimizer_updates=0); return
    result=dict(identity=identity,complete=True,trials=old+trials,summary=summaries(old+trials,reg),
        new_fits=27,cached_verified_fits=9,total_new_fit_steps=sum(t['fit']['step'] for t in trials),
        summed_new_fit_seconds=sum(t['fit']['fit_seconds'] for t in trials),
        primary='past_normalized_ADE_equal_physical_scene',independent_confirmation=False,
        main_primary_changed=False,sealed_roles_opened=False,new_deployment=False,stage5c_executed=False,smc_enabled=False)
    report_files(result,reports)
    heartbeat(state='complete',trials=27,new_optimizer_updates=updates)


if __name__=='__main__':
    main()
