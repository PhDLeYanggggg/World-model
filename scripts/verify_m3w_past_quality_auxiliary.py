"""Independent reduction of fixed past-quality versus placebo reports."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_past_quality_auxiliary_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def compare(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys();return sum(compare(a[k],b[k]) for k in a)
    if isinstance(a,list):
        assert len(a)==len(b);return sum(compare(x,y) for x,y in zip(a,b))
    if isinstance(a,float):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-9),(a,b)
    else:assert a==b,(a,b)
    return 1


def ci(rows,key,draws,seed):
    sites={}
    for row in rows:
        v=row['result']['contrasts'][key]
        if v is not None:sites.setdefault(row['source'],[]).append(v)
    if not sites:return dict(mean=None,CI95=None,localities=0)
    v=[math.fsum(sites[s])/len(sites[s]) for s in sorted(sites)]
    ix=np.random.default_rng(seed).integers(0,len(v),(draws,len(v)))
    boot=[math.fsum(v[int(i)] for i in sample)/len(v) for sample in ix]
    return dict(mean=math.fsum(v)/len(v),CI95=np.quantile(boot,[.025,.975]).tolist(),
                localities=len(v),nominal_exposed_development_only=True)


def aggregate(rows,cfg):
    out=dict(groups=len(rows),auxiliary_fits=len(rows)*2,new_tree_splits=0,new_neural_updates=0,
             independent_confirmation=False,transfer_evaluated=False,deployment_changed=False)
    for name in rows[0]['result']['policies']:
        values=[r['result']['policies'][name] for r in rows]
        easy=[v['easy_selected_risk_upper'] for v in values if v['easy_selected_risk_upper'] is not None]
        out[name]=dict(selected=sum(v['selected_count'] for v in values),
            unknown_selected=sum(v['selected_unknown'] for v in values),
            complete_support=sum(v['finite_completion_supported'] for v in values),defined_easy_risk=len(easy),
            violations=sum(v>.02+1e-12 for v in easy),worst_easy_upper=max(easy) if easy else None)
    for k in rows[0]['result']['contrasts']:out[k]=ci(rows,k,cfg['bootstrap_resamples'],cfg['bootstrap_seed'])
    reasons=[]
    for other in ('original','placebo'):
        if out['quality_minus_'+other+'_signed_MSE']['CI95'][1]>=0:reasons.append('MSE_not_supported_vs_'+other)
        for mode in ('full','matched'):
            if out['quality_minus_'+other+'_'+mode+'_utility_percent']['CI95'][0]<=0:
                reasons.append(mode+'_utility_not_supported_vs_'+other)
    q,o=out['quality'],out['original']
    if q['complete_support']<o['complete_support']:reasons.append('complete_support_reduced')
    if q['violations']>o['violations']:reasons.append('more_risk_violations')
    if q['worst_easy_upper'] is None or o['worst_easy_upper'] is None or q['worst_easy_upper']>o['worst_easy_upper']:
        reasons.append('worst_risk_not_preserved')
    out['advance_to_transfer']=not reasons
    return out,reasons


def main():
    complete=json.loads((PUBLIC/'complete.json').read_text());summary=json.loads((PUBLIC/'summary.json').read_text())
    assert sha(PUBLIC/'summary.json')==complete['summary_sha256']
    reg=json.loads((PUBLIC/'registration.json').read_text())
    for p,h in reg['bindings'].items():assert sha(ROOT/p)==h
    cfg=json.loads((ROOT/'configs/m3w_european_past_quality_auxiliary_v1.json').read_text())
    rows=[];checks=0
    for ref in complete['groups']:
        assert sha(ROOT/ref['path'])==ref['sha256'];r=json.loads((ROOT/ref['path']).read_text());rows.append(r)
        for arm in ('quality','placebo'):
            fit=r['training'][arm]
            assert fit['mean_leaf_training_loss_after']<=fit['mean_leaf_training_loss_before']+1e-9
            assert fit['training_zero_mean_max']<1e-9
        assert r['training']['quality']['past_quality_hash']==r['training']['placebo']['past_quality_hash']
        for other in ('original','placebo'):
            for mode in ('full','matched'):
                a=r['result']['policies'][other if mode=='full' else other+'_matched_quality']
                b=r['result']['policies']['quality' if mode=='full' else 'quality_matched_'+other]
                d=100*(b['selected_net_gain_lower_mass']-a['selected_net_gain_lower_mass'])/r['result']['full_known_reference_mass']
                checks+=compare(d,r['result']['contrasts']['quality_minus_'+other+'_'+mode+'_utility_percent'])
                if mode=='matched':assert a['selected_count']==b['selected_count']
        checks+=compare(r['result']['scores']['quality']-r['result']['scores']['original'],r['result']['contrasts']['quality_minus_original_signed_MSE'])
    assert len(rows)==72 and len({r['source'] for r in rows})==12
    independent,reasons=aggregate(rows,cfg);checks+=compare(independent,summary)
    o=summary['original'];assert (o['selected'],o['unknown_selected'],o['complete_support'],o['violations'])==(95455,918,33,7)
    manifest=json.loads((PUBLIC/'checkpoint_manifest.json').read_text())
    assert len(manifest['checkpoints'])==144
    expected=sorted((cp['group'],cp['sha256'],cp['bytes']) for r in rows for cp in r['checkpoints'].values())
    actual=sorted((cp['group'],cp['sha256'],cp['bytes']) for cp in manifest['checkpoints'])
    assert expected==actual and sum(cp['bytes'] for cp in manifest['checkpoints'])==complete['checkpoint_bytes']
    receipt=dict(independent_scalar_and_bootstrap_fields_checked=checks,source_heads=72,auxiliary_fits=144,
        parent_count_anchors_checked=4,gate_failure_reasons=reasons,
        summary_sha256=sha(PUBLIC/'summary.json'),complete_sha256=sha(PUBLIC/'complete.json'),
        raw_parameter_refit_replayed=complete['exact_fit_replay'],inference_replayed=complete['exact_inference_replay'],
        checkpoint_manifest_sha256=sha(PUBLIC/'checkpoint_manifest.json'),independent_confirmation=False)
    dest=PUBLIC/'verification.json';payload=json.dumps(receipt,indent=2)+'\n'
    if dest.exists():assert dest.read_text()==payload
    else:dest.write_text(payload)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
