"""Fixed cross-objective harm review; freeze decisions before outcome access."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
import time
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_conditional_cost as parent
from scripts import run_m3w_bounded_cost as bounded
from scripts.run_m3w_tempered_cost import causal_scores
from scripts.run_m3w_native_forecast import assert_current, immutable_json, json_write
from scripts.run_m3w_native_nested import write_arrays
from src.evaluation.m3w_cross_objective_review import choices, empirical_gate, POLICIES
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
from src.evaluation.m3w_native_matched_coverage import paired_scene_contrast
from src.evaluation.m3w_forecast_cost_bounds import partial_gain_bounds
import numpy as np
import torch

CONFIG = 'configs/m3w_cross_objective_review_v1.json'
CODE = ('scripts/run_m3w_cross_objective_review.py', 'scripts/verify_m3w_cross_objective_review.py',
        'src/evaluation/m3w_cross_objective_review.py', 'tests/test_m3w_cross_objective_review.py')


def validate_config(cfg, pcfg):
    if (cfg['sites'] != pcfg['sites'] or cfg['seeds'] != pcfg['seeds']
            or cfg['nomination'] != 'frozen_region_strict_stop' or cfg['reviewers'] != ['native', 'fraction']
            or cfg['harm_ratio'] != .1 or cfg['policies'] != list(POLICIES)
            or cfg['primary_contrast'] != 'reviewed_minus_matched_nomination'
            or cfg['bootstrap_resamples'] != 3000 or any(cfg[k] for k in (
                'threshold_search', 'model_selection', 'new_training', 'risk_calibration',
                'closed_role_readout', 'independent_confirmation', 'deployment',
                'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed review and equal-count comparison only')


def load():
    cfg = json.loads((ROOT/CONFIG).read_text())
    pcfg,data,views,predictions,budget,refs,pid = parent.load()
    records,nomination = parent.load_states(pcfg,views,pid)
    validate_config(cfg,pcfg)
    prior = json.loads((ROOT/cfg['parent_analysis']).read_text())
    assert prior['identity'] == pid
    bindings = dict(pid['source_bindings'])
    def bind(path, expected=None):
        sha = file_digest(ROOT/path)
        if (expected is not None and expected != sha) or (path in bindings and bindings[path] != sha):
            raise ValueError('Changed evidence '+path)
        bindings[path] = sha
    for path in (CONFIG,cfg['registration'],cfg['parent_analysis'],cfg['motivation'],*CODE):
        bind(path)
    for source in (cfg['parent_analysis'],cfg['motivation']):
        for name in ('replay.json','independent_verification.json'):
            path = str(Path(source).parent/name);r=json.loads((ROOT/path).read_text())
            assert r['all_checks_passed'] and r['analysis_sha256'] == bindings[source]
            bind(path)
    for r in prior['archives']:
        bind(r['path'],r['sha256'])
    for r in records.values():
        bind(r['checkpoint'],r['checkpoint_sha256'])
    identity = dict(source_bindings=bindings,config=cfg,parent_identity=pid,
        runtime=dict(torch_threads=4,interop_threads=1,num_workers=0),
        numpy=np.__version__,torch=torch.__version__,architecture=platform.machine(),
        new_training=False,all_four_sites_design_exposed=True)
    assert_current(identity)
    return cfg,pcfg,data,views,predictions,budget,prior,refs,nomination,identity


def evaluate(cfg,pcfg,data,views,predictions,budget,prior,refs,nomination,identity,beat,verify=False):
    root,public = ROOT/cfg['output'],ROOT/cfg['reports'];n=len(data['sites'])
    old = {r['view']:r for r in prior['archives']}
    controls = {r['view']:r for r in budget['archives']}
    chosen = {s:{p:np.zeros(n,bool) for p in POLICIES} for s in cfg['seeds']}
    candidates = {s:np.empty((n,12,2),np.float32) for s in cfg['seeds']}
    archives,scores = [],{}
    for key,meta in views.items():
        with np.load(ROOT/predictions[key]['path'],allow_pickle=False) as z:
            ids,prediction=z['ids'].copy(),z['prediction'].copy()
        np.testing.assert_array_equal(ids,np.flatnonzero(data['sites']==meta['outer_site']))
        with np.load(ROOT/old[key]['path'],allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],ids)
            a,d,original=z['score'].copy(),z['distance'].copy(),z['strict_stop'].copy()
        with np.load(ROOT/controls[key]['path'],allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],ids)
            np.testing.assert_array_equal(z['distance'],d)
            b,c=z['native_score'].copy(),z['fraction_score'].copy()
        if verify:
            for recorded,cp in ((a,nomination[key]),(b,refs[key,'native']),(c,refs[key,'fraction'])):
                fresh,dd=causal_scores(data,ids,prediction,cp,pcfg)
                np.testing.assert_array_equal(recorded,fresh);np.testing.assert_array_equal(d,dd)
        bits=choices(a,b,c,data['geometry'][ids,:16].reshape(-1,8,2),d,ids)
        np.testing.assert_array_equal(bits['nomination'],original)
        assert bits['reviewed'].sum()==bits['matched_nomination'].sum()
        for p,use in bits.items():chosen[meta['seed']][p][ids]=use
        candidates[meta['seed']][ids]=prediction;scores[key]=(a,b,c)
        path=root/'decisions'/f'{key}.npz'
        if verify and not path.exists():raise ValueError('Missing decisions cannot be replayed')
        write_arrays(path,dict(ids=ids,nomination_score=a,native_score=b,fraction_score=c,distance=d,**bits))
        archives.append(dict(view=key,path=str(path.relative_to(ROOT)),sha256=file_digest(path)))
        beat(state='head_scores_replayed' if verify else 'decisions_frozen',view=key)
    immutable_json(root/'decisions_complete.json',dict(identity=identity,archives=archives,
        future_targets_used_in_decisions=False,decision_rows=n*len(cfg['seeds'])))
    y,valid=bounded.read_arrays(data,np.arange(n),'target'),bounded.read_arrays(data,np.arange(n),'valid')
    baseline=data['geometry'][:,332:356].reshape(n,12,2)
    cv,cf=native_errors(baseline,y,valid,data['scale']);full=valid.all(1)
    masks=dict(complete=full,zero_CV=full&(cv==0),hard=np.zeros(n,bool),positive_easy=np.zeros(n,bool))
    errors={s:native_errors(v,y,valid,data['scale']) for s,v in candidates.items()}
    bounds={s:partial_gain_bounds(v,baseline,y,valid,data['scale']) for s,v in candidates.items()}
    quality=[]
    for key,meta in views.items():
        ids=np.flatnonzero(data['sites']==meta['outer_site']);pr=nomination[key]['preprocess']
        masks['hard'][ids]=cv[ids]>=pr['hard_cut']
        masks['positive_easy'][ids]=(cv[ids]>0)&(cv[ids]<=pr['positive_easy_cut'])
        delta=cv[ids]-errors[meta['seed']][0][ids]
        a,b,c=scores[key];h=np.maximum(-delta,0);rh=np.maximum.reduce((a[:,1],b[:,1],c[:,1]))
        bits={p:chosen[meta['seed']][p][ids] for p in POLICIES}
        for name,use in dict(**bits,rejected=bits['nomination']&~bits['reviewed']).items():
            complete=use&full[ids]
            quality.append(dict(view=key,group=name,rows=int(use.sum()),complete=int(complete.sum()),
                costs=None if not complete.any() else dict(realized_harm=float(h[complete].mean()),
                    realized_net_gain=float(delta[complete].mean()),
                    nomination_harm=float(a[complete,1].mean()),review_harm=float(rh[complete].mean()))))
    def metric(a,r,mask=None):
        if mask is None:mask=np.ones(n,bool)
        return paired_scene_metrics(a[mask],r[mask],data['sites'][mask],expected_scenes=cfg['sites'],
            dataset='sdd',coordinate_unit='annotation_pixel',bootstrap_resamples=cfg['bootstrap_resamples'])
    summaries={}
    for policy in POLICIES:
        seeds,ades,fdes={},[],[]
        for seed in cfg['seeds']:
            use=chosen[seed][policy];ade=np.where(use,errors[seed][0],cv);fde=np.where(use,errors[seed][1],cf)
            ades.append(ade);fdes.append(fde)
            seeds[str(seed)]=dict(ADE=metric(ade,cv),FDE=metric(fde,cf),selected=int(use.sum()),
                selected_unknown=int((use&~valid.any(1)).sum()),selected_incomplete=int((use&~full).sum()),
                zero_CV_harmed=int((ade[masks['zero_CV']]>0).sum()),
                full_grid_absolute_gain_bounds={s:[float(np.where(use,bounds[seed][k],0)[data['sites']==s].mean())
                    for k in ('lower','upper')] for s in cfg['sites']},
                subsets={g:metric(ade,cv,m) for g,m in masks.items()})
        summaries[policy]=dict(ADE=metric(np.mean(ades,0),cv),FDE=metric(np.mean(fdes,0),cf),seeds=seeds,
            subsets={g:metric(np.mean(ades,0),cv,m) for g,m in masks.items()})
    assert summaries['nomination']==prior['summaries']['strict_stop']
    contrasts={p:paired_scene_contrast(
        [summaries['reviewed']['ADE']['by_scene'][s]['gain_percent'] for s in cfg['sites']],
        [summaries[p]['ADE']['by_scene'][s]['gain_percent'] for s in cfg['sites']])
        for p in ('nomination','matched_nomination')}
    gates=empirical_gate(summaries['reviewed'],contrasts['matched_nomination'])
    result=dict(identity=identity,result_source='fresh_fixed_policy_readout_from_cached_verified_heads',
        summaries=summaries,contrasts=contrasts,conditional_quality=quality,archives=archives,
        primary_gates=gates,primary_joint_empirical_pass=all(gates.values()),new_training=False,
        model_selection=False,risk_calibrated=False,independent_confirmation=False,deployment=False,
        stage5c_executed=False,smc_enabled=False,decision_manifest_sha256=file_digest(root/'decisions_complete.json'))
    assert_current(identity);immutable_json(public/'analysis.json',result)
    if verify:immutable_json(public/'replay.json',dict(analysis_sha256=file_digest(public/'analysis.json'),
        all_checks_passed=True,checkpoint_endpoints_replayed=36,score_rows=n*len(cfg['seeds'])*3))
    beat(state='verified' if verify else 'evaluated',gates=gates)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    for name in ('audit-only','evaluate','verify'):g.add_argument('--'+name,action='store_true')
    args=p.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    cfg=json.loads((ROOT/CONFIG).read_text());root=ROOT/cfg['output'];root.mkdir(parents=True,exist_ok=True)
    def beat(**v):
        event=dict(pid=os.getpid(),timestamp_unix=time.time(),**v);json_write(root/'heartbeat.json',event)
        with (root/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        payload=load();identity=payload[-1];immutable_json(root/'identity.json',identity)
        if args.audit_only:beat(state='preflight_pass',bindings=len(identity['source_bindings']),new_training=False)
        else:evaluate(*payload,beat,args.verify)


if __name__=='__main__':main()
