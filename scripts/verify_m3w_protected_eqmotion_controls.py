"""Independent choices, costs and scene arithmetic for the EqMotion extension."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_protected_eqmotion_controls import load
from scripts.run_m3w_bounded_cost import read_arrays, training_data
from scripts.run_m3w_native_forecast import array_hash, file_digest, assert_current, immutable_json
from scripts.verify_m3w_native_joint_controls import distances
from scripts.verify_m3w_protected_motion_controls import check_metrics
from src.data_unification.m3w_causal_recordings import BASELINES
import joblib
import numpy as np
import torch


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg,data,views,predictions,mv,neural,refs,ma,_,cuts,identity = load()
    public, root = ROOT/cfg['reports'], ROOT/cfg['output']
    ap = public/'analysis.json'
    analysis, replay = json.loads(ap.read_text()), json.loads((public/'replay.json').read_text())
    assert analysis['identity'] == identity and replay['all_checks_passed'] and replay['analysis_sha256'] == file_digest(ap)
    receipt = json.loads((root/'decisions_complete.json').read_text())
    assert receipt['all_forests_complete'] and not receipt['target_arrays_loaded_for_decisions']
    n = len(data['sites'])
    y, mask = read_arrays(data, np.arange(n), 'target'), read_arrays(data, np.arange(n), 'valid')
    base = data['geometry'][:, 332:356].reshape(n, 12, 2)
    cv, cf = distances(base, y, mask, data['scale'])
    subsets = dict(complete=mask.all(1), zero_CV=mask.all(1) & (cv == 0),
        hard=np.zeros(n, bool), positive_easy=np.zeros(n, bool))
    actions = ['eqmotion']+cfg['comparators']
    costs, ends, lower, upper = ({(s,a):np.empty(n) for s in cfg['seeds'] for a in actions} for _ in range(4))
    choices, matched = {}, {}
    for s in cfg['seeds']:
        for a in actions:
            for h in cfg['heads']:
                for p in cfg['policies']:
                    choices[s,a,h,p] = np.zeros(n,bool)
                if a != 'eqmotion':
                    for side in ('eqmotion','comparator'): matched[s,a,h,side] = np.zeros(n,bool)
    policy_checks, matched_checks, budgets = 0,0,0
    for key, meta in views.items():
        seed, held = meta['seed'], meta['outer_site']
        ids, train = np.flatnonzero(data['sites'] == held), np.flatnonzero(data['sites'] != held)
        stored = next(r for r in analysis['decision_archives'] if r['view'] == key)
        assert file_digest(ROOT/stored['path']) == stored['sha256']
        with np.load(ROOT/stored['path'],allow_pickle=False) as z: q = {k:z[k].copy() for k in z.files}
        np.testing.assert_array_equal(ids,q['ids'])
        ctrl = next(r for r in ma['decision_archives'] if r['view'] == key)
        with np.load(ROOT/ctrl['path'],allow_pickle=False) as z: old = {k:z[k].copy() for k in z.files}
        with np.load(ROOT/predictions[key]['path'],allow_pickle=False) as z: ep = z['prediction'].copy()
        with np.load(ROOT/mv[key]['outer_prediction']['prediction']['path'],allow_pickle=False) as z: tp = z['prediction'].copy()
        pr = torch.load(ROOT/neural[key]['checkpoint'],map_location='cpu',weights_only=False)
        ref = torch.load(ROOT/refs[key]['checkpoint'],map_location='cpu',weights_only=False)
        f = json.loads((root/'trials'/key/'complete.json').read_text())
        assert file_digest(ROOT/f['checkpoint']) == f['checkpoint_sha256']
        forest = joblib.load(ROOT/f['checkpoint'])
        ix,x,labels,d,p = training_data(meta,data)
        np.testing.assert_array_equal(ix,train)
        assert forest['identity']['inputs_sha256'] == array_hash(train,x,d)
        assert forest['identity']['labels_sha256'] == array_hash(labels)
        for cp in (pr,forest):
            np.testing.assert_array_equal(cp['draws'],ref['draws'])
            np.testing.assert_array_equal(cp['preprocess']['known'],mask[train].all(1))
            assert cp['preprocess']['training_sites'] == sorted(set(cfg['sites'])-{held})
            for field in ('mean','std','known','weights','constant'):
                np.testing.assert_array_equal(cp['preprocess'][field],p[field])
            assert cp['preprocess']['cost_scale'] == p['cost_scale']
        assert len(forest['model'].estimators_) == 128 and pr['step'] == 3000
        np.testing.assert_array_equal(forest['sample_weight'],pr['draws']*(d>0))
        budgets += 2
        subsets['hard'][ids] = cv[ids] >= cuts[key]['hard_cut']
        subsets['positive_easy'][ids] = (cv[ids] > 0) & (cv[ids] <= cuts[key]['positive_easy_cut'])
        past = data['geometry'][ids,:16].reshape(-1,8,2)
        for action in actions:
            k = None if action in ('eqmotion','transformer') else BASELINES.index(action)
            pred = ep if action=='eqmotion' else tp if action=='transformer' else data['geometry'][ids,308+24*k:332+24*k].reshape(-1,12,2)
            costs[seed,action][ids],ends[seed,action][ids] = distances(pred,y[ids],mask[ids],data['scale'][ids])
            separation = np.sqrt(np.sum((pred.astype(float)-base[ids].astype(float))**2,axis=-1))*data['scale'][ids,None]
            delta = np.sqrt(np.sum((base[ids].astype(float)-y[ids].astype(float))**2,axis=-1))
            delta -= np.sqrt(np.sum((pred.astype(float)-y[ids].astype(float))**2,axis=-1))
            observed = np.where(mask[ids],delta*data['scale'][ids,None],0).sum(1)/12
            radius = np.where(mask[ids],0,separation).sum(1)/12
            lower[seed,action][ids],upper[seed,action][ids] = observed-radius,observed+radius
            for head in cfg['heads']:
                score = q[head+'_score'] if action=='eqmotion' else old[action+'__'+head+'__score']
                assert np.isfinite(score).all() and (score>=0).all()
                assert np.all(score.sum(1)<=separation.mean(1)+2e-6*(1+separation.mean(1)))
                net = (score[:,0]>score[:,1]) & (separation.mean(1)>0) & np.any(past[:,-1]!=past[:,-2],axis=1)
                strict = net & (score[:,1]<=.1*score[:,0])
                for policy,selected in [('net_stop',net),('strict_stop',strict)]:
                    expected = q[head+'_'+policy] if action=='eqmotion' else old[action+'__'+head+'__'+policy]
                    np.testing.assert_array_equal(selected,expected)
                    choices[seed,action,head,policy][ids] = selected
                    policy_checks += 1
        for action in cfg['comparators']:
            for head in cfg['heads']:
                count = min(q[head+'_strict_stop'].sum(),old[action+'__'+head+'__strict_stop'].sum())
                for side,score,pool in [('eqmotion',q[head+'_score'],q[head+'_strict_stop']),
                        ('comparator',old[action+'__'+head+'__score'],old[action+'__'+head+'__strict_stop'])]:
                    ranked=sorted(np.flatnonzero(pool),key=lambda i:(float(score[i,1]-score[i,0]),int(ids[i])))
                    use=np.zeros(len(ids),bool); use[ranked[:count]]=True
                    np.testing.assert_array_equal(use,q[action+'__'+head+'__matched_'+side])
                    matched[seed,action,head,side][ids]=use
                matched_checks+=1
        print(json.dumps(dict(state='view_verified',view=key,budgets=budgets)),flush=True)
    reductions=0
    def verify_summary(action,selected,report):
        nonlocal reductions
        aa,ff=[],[]
        for seed in cfg['seeds']:
            use=selected[seed]; a=np.where(use,costs[seed,action],cv); f=np.where(use,ends[seed,action],cf)
            aa.append(a);ff.append(f)
            r=report['seeds'][str(seed)]
            assert r['selected']==use.sum() and r['selected_unknown']==(use & ~mask.any(1)).sum()
            assert r['selected_incomplete']==(use & ~mask.all(1)).sum()
            assert r['zero_CV_harmed']==(a[subsets['zero_CV']]>0).sum()
            reductions+=check_metrics(a,cv,data['sites'],cfg['sites'],r['ADE'])
            reductions+=check_metrics(f,cf,data['sites'],cfg['sites'],r['FDE'])
            for group,m in subsets.items(): reductions+=check_metrics(a[m],cv[m],data['sites'][m],cfg['sites'],r['subsets'][group])
            for site in cfg['sites']:
                bound=[np.where(use,b[seed,action],0)[data['sites']==site].mean() for b in (lower,upper)]
                np.testing.assert_allclose(bound,r['full_grid_gain_bounds'][site],rtol=1e-10,atol=1e-10)
        reductions+=check_metrics(np.mean(aa,0),cv,data['sites'],cfg['sites'],report['ADE'])
        reductions+=check_metrics(np.mean(ff,0),cf,data['sites'],cfg['sites'],report['FDE'])
        for group,m in subsets.items(): reductions+=check_metrics(np.mean(aa,0)[m],cv[m],data['sites'][m],cfg['sites'],report['subsets'][group])
    for name, r in analysis['summaries'].items():
        selected=({s:np.ones(n,bool) for s in cfg['seeds']} if name=='eqmotion__uncontrolled' else
                  {s:choices[s,'eqmotion',*name.split('__')[1:]] for s in cfg['seeds']})
        verify_summary('eqmotion',selected,r)
    for key,r in analysis['comparisons'].items():
        action,head=key.split('__')
        verify_summary(action,{s:choices[s,action,head,'strict_stop'] for s in cfg['seeds']},r['comparator'])
        for side in ('eqmotion','comparator'):
            verify_summary('eqmotion' if side=='eqmotion' else action,
                {s:matched[s,action,head,side] for s in cfg['seeds']},analysis['matched_count'][key][side])
    out=dict(all_checks_passed=True,analysis_sha256=file_digest(ap),verifier_sha256=file_digest(Path(__file__)),
        shared_independent_reducer_sha256=file_digest(ROOT/'scripts/verify_m3w_protected_motion_controls.py'),
        matched_model_budgets_checked=budgets,policy_choices_checked=policy_checks,matched_pairs_checked=matched_checks,
        scene_reductions=reductions,independent_arithmetic_same_agent=True,independent_research_confirmation=False,
        external_readout=False,deployment=False)
    assert_current(identity);immutable_json(public/'independent_verification.json',out)
    print(json.dumps(out,indent=2))


if __name__=='__main__': main()
