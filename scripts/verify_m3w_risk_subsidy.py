"""Separate arithmetic and exhaustive small-query checks, not external validation."""
import csv
import itertools
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_m3w_risk_subsidy import load, NEW, MATCHED
from scripts.run_m3w_native_forecast import file_digest, immutable_json, assert_current
import numpy as np
import torch


def raw_ok(q,r,b,mode):
    risk=np.maximum(q,0) if 'clipped' in mode else q
    budget=.02*math.fsum(r[b] if mode.endswith('_selected') else r)
    return math.fsum(risk[b]) <= budget


def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    cfg,data,parent,old,pdm,identity=load();root=ROOT/cfg['output'];public=ROOT/cfg['reports']
    analysis=json.loads((public/'analysis.json').read_text());dm=json.loads((root/'decisions_complete.json').read_text())
    assert analysis['experiment_sha256']==file_digest(root/'identity.json')==dm['experiment_sha256']
    assert analysis['decision_manifest_sha256']==file_digest(root/'decisions_complete.json')
    sources={r['sha256']:r for r in pdm['receipts']};n=len(data['sites']);arms=cfg['policies']
    groups={(s,a):dict(bits=np.zeros((n,len(arms)),bool),seen=np.zeros(n,bool)) for s in cfg['seeds'] for a in cfg['actions']}
    checks=0;optima=0;skipped=0;slots=set();nonoptimal=0;countfail=0;ref_failed=0;closed=0;skipped_uncertified=0
    for ref in dm['receipts']:
        assert file_digest(ROOT/ref['path'])==ref['sha256'];record=json.loads((ROOT/ref['path']).read_text())
        assert record['experiment_sha256']==dm['experiment_sha256']
        assert file_digest(ROOT/record['path'])==record['sha256']
        sr=sources[record['source_receipt_sha256']];source=json.loads((ROOT/sr['path']).read_text())
        assert source['view']==record['view'] and source['action']==record['action']
        with np.load(ROOT/source['path'],allow_pickle=False) as z: z={k:z[k].copy() for k in z.files}
        with np.load(ROOT/record['path'],allow_pickle=False) as v:
            ids=v['ids'].copy();bits=v['choices'].copy()
        np.testing.assert_array_equal(ids,z['ids']); key=int(record['view'].rsplit('_seed',1)[1]),record['action']
        group=groups[key];assert not group['seen'][ids].any();group['seen'][ids]=True;group['bits'][ids]=bits
        for j,mode in enumerate(arms[:4]):np.testing.assert_array_equal(bits[:,j],z['choices'][:,parent['policies'].index(mode.removeprefix('parent_'))])
        for qr in record['queries']:
            rows=np.flatnonzero((data['recordings'][ids]==qr['recording'])&(data['frames'][ids]==qr['frame']))
            assert len(rows)
            q,r,g,ok=[z[k][rows] for k in ('net_risk','denominator','gain','support')]
            local=bits[rows];target=int(local[:,arms.index('net_clipped_selected')].sum())
            for mode,info in qr['arms'].items():
                b=local[:,arms.index(mode)];assert not (b&~ok).any();assert info['selected']==b.sum()
                assert raw_ok(q,r,b,mode) and info['direct_constraint_pass']
                if mode.startswith('matched_'):
                    assert info['requested_count']==target
                    assert info['exact_count_pass']==bool(b.sum()==target)
                    assert info['reference_optimal']==qr['arms']['net_clipped_selected']['optimal']
                    assert info['reference_failed_closed']==qr['arms']['net_clipped_selected']['failed_closed']
                    ref_failed+=info['reference_failed_closed'];countfail+=not info['exact_count_pass']
                if info['failed_closed']:assert not b.any();closed+=1
                nonoptimal+=not info['canonical_optimal_verified']
                checks+=1
            # Feasible-set nesting concerns predicted objective, not observed ADE.
            refgain=math.fsum(g[local[:,arms.index('net_clipped_selected')]])
            if qr['arms']['net_clipped_selected']['optimal']:
                for mode in (*NEW,*MATCHED):
                    if qr['arms'][mode]['optimal']:
                        assert math.fsum(g[local[:,arms.index(mode)]])>=refgain-1e-8*(1+abs(refgain))
            slot=(record['view'],record['action'],qr['recording'])
            if slot not in slots:
                slots.add(slot)
                if ok.sum()>16:skipped+=1;continue
                active=np.flatnonzero(ok);options=[]
                for v in itertools.product([False,True],repeat=len(active)):
                    b=np.zeros(len(ok),bool);b[active]=v;options.append(b)
                gain=np.array([math.fsum(g[b]) for b in options])
                for mode,info in qr['arms'].items():
                    if not info['optimal']:skipped_uncertified+=1;continue
                    valid=np.array([raw_ok(q,r,b,mode) and (not mode.startswith('matched_') or b.sum()==target) for b in options])
                    assert valid.any()
                    np.testing.assert_allclose(math.fsum(g[local[:,arms.index(mode)]]),gain[valid].max(),rtol=1e-7,atol=1e-9)
                    optima+=1
    assert all(v['seen'].all() for v in groups.values())
    counts=0;contrasts=0;sites={};rows=[]
    for action in cfg['actions']:
        for seed in cfg['seeds']:
            ref=next(v for v in old['outcome_archives'] if v['path'].endswith(f'{action}_seed{seed}.npz'))
            with np.load(ROOT/ref['path'],allow_pickle=False) as z:o={k:z[k].copy() for k in z.files}
            for j,mode in enumerate(arms):
                b=groups[seed,action]['bits'][:,j];e=np.where(b,o['candidate_ade'],o['cv']);f=np.where(b,o['candidate_fde'],o['cf'])
                sm=analysis['summary'][action+'__'+mode]['seeds'][str(seed)]
                assert sm['selected']==b.sum();assert sm['zero_CV_harmed']==int((b&o['zero_CV']&(e>0)).sum())
                assert sm['selected_incomplete']==int((b&~o['complete']).sum())
                for subset in ('all','hard','positive_easy','complete','zero_CV'):
                    mask=np.ones(n,bool) if subset=='all' else o[subset]
                    summary=sm['ADE'] if subset=='all' else sm['subsets'][subset]
                    for site in cfg['sites']:
                        use=mask&(data['sites']==site)&np.isfinite(o['cv']);r=summary['by_scene'][site]
                        assert r['rows']==use.sum()
                        if use.any():
                            np.testing.assert_allclose(r['model_error'],e[use].mean(),rtol=1e-12,atol=1e-12)
                            np.testing.assert_allclose(r['reference_error'],o['cv'][use].mean(),rtol=1e-12,atol=1e-12)
                            if o['cv'][use].sum()>0:
                                gain=100*(1-e[use].sum()/o['cv'][use].sum())
                                np.testing.assert_allclose(r['gain_percent'],gain,atol=1e-10)
                                sites.setdefault((action,mode,subset,site),[]).append(gain)
                        rows.append(dict(action=action,policy=mode,seed=seed,site=site,subset=subset,rows=r['rows'],
                            gain_percent=r['gain_percent'],absolute_harm=r['absolute_harm']));counts+=1
                for site in cfg['sites']:
                    use=(data['sites']==site)&np.isfinite(o['cf'])
                    np.testing.assert_allclose(sm['FDE']['by_scene'][site]['model_error'],f[use].mean(),rtol=1e-12)
        for pair,details in analysis['contrasts'][action].items():
            left,right=pair.split('_minus_')
            for subset,value in details.items():
                delta=np.array([np.mean(sites[action,left,subset,s])-np.mean(sites[action,right,subset,s]) for s in cfg['sites']])
                boot=np.random.default_rng(38113).choice(delta,size=(3000,4)).mean(1)
                np.testing.assert_allclose(value['mean_gain_difference_pp'],delta.mean(),atol=1e-10)
                np.testing.assert_allclose(value['ci95_pp'],np.quantile(boot,[.025,.975]),atol=1e-10);contrasts+=1
    assert_current(identity)
    result=dict(all_checks_passed=True,analysis_sha256=file_digest(public/'analysis.json'),
        verifier_sha256=file_digest(Path(__file__)),query_constraints=checks,small_query_optima=optima,
        exhaustive_skipped_large=skipped,scene_reductions=counts,paired_contrasts=contrasts,
        canonical_optimum_unverified=nonoptimal,failed_closed=closed,
        count_failures=countfail,matched_failed_reference_instances=ref_failed,
        exhaustive_skipped_uncertified=skipped_uncertified,
        same_agent_separate_arithmetic=True,independent_research_confirmation=False)
    immutable_json(public/'independent_arithmetic.json',result)
    compact=[]
    for key,sm in analysis['summary'].items():
        action,mode=key.split('__')
        easy=[r['gain_percent'] for s in sm['seeds'].values() for r in s['subsets']['positive_easy']['by_scene'].values() if r['gain_percent'] is not None]
        compact.append(dict(action=action,policy=mode,ADE_gain=sm['ADE']['equal_scene_gain_percent'],
            CI=sm['ADE']['scene_bootstrap_ci95'],FDE_gain=sm['FDE']['equal_scene_gain_percent'],
            hard_gain=sm['subsets']['hard']['equal_scene_gain_percent'],worst_easy_degradation=max(0.,-min(easy)),
            switch_rate=np.mean([v['selected']/n for v in sm['seeds'].values()]),
            zero_CV_harmed=sum(v['zero_CV_harmed'] for v in sm['seeds'].values()),
            unknown_selected=sum(v['selected_unknown'] for v in sm['seeds'].values()),
            incomplete_selected=sum(v['selected_incomplete'] for v in sm['seeds'].values())))
    immutable_json(public/'compact_results.json',dict(rows=compact,analysis_sha256=file_digest(public/'analysis.json')))
    for name,values in [('results.csv',compact),('site_seed_results.csv',rows)]:
        with (public/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]),lineterminator='\n');w.writeheader();w.writerows(values)
    text=['# Query Risk Credit and Denominator Results','','All fixed controls; no winner selected or deployed.',
        'Obs8/pred12 native SDD pixels, four development-exposed sites, three seeds. Not historical t+50.',
        'CI: 3000 physical-site bootstrap draws, not independent confirmation or calibrated safety.','',
        '| Predictor / rule | ADE gain % | CI95 | Hard gain % | Worst easy degradation % | Switch % | Zero-CV harmed |',
        '|---|---:|---|---:|---:|---:|---:|']
    for r in compact:
        text.append(f"| {r['action']} / {r['policy']} | {r['ADE_gain']:.6f} | {r['CI']} | {r['hard_gain']:.6f} | {r['worst_easy_degradation']:.6f} | {100*r['switch_rate']:.4f} | {r['zero_CV_harmed']} |")
    text+=['','Missing labels are unknown. Solver failures, failed-reference matched queries, tails, partial-label bounds',
        'and full contrasts remain in analysis.json. Signed and clipped-net risk are not positive-harm guarantees.']
    (public/'results.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
