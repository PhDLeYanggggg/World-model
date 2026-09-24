"""Separate SciPy distance arithmetic for every registered rare-case query."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import audit_m3w_moving_zero_support as run
import joblib
import numpy as np
import torch
from scipy.spatial.distance import cdist


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg,ctx,_,_,_,identity=run.load(); root=ROOT/cfg['output']; public=ROOT/cfg['reports']
    assert run.parent.read(root/'identity.json')==identity
    a=run.parent.read(public/'analysis.json')
    assert a['identity_sha256']==run.base.file_digest(root/'identity.json')
    data=ctx['parent']['pack'][1][1]; checked=0; max_delta=0.; files=0
    hc=np.ravel(np.column_stack((np.arange(8)*3,np.arange(8)*3+1)))
    for ref in a['views']:
        assert run.base.file_digest(ROOT/ref['path'])==ref['sha256']; files+=1
        r=run.parent.read(ROOT/ref['path']); cases=[v for v in r['records'] if v['zero_case']]
        if not cases: continue
        key,action=r['view'],r['action']; pc=ctx['parent']
        ids,raw,d,pr,q,draws,scale=run.base.prepare(pc['pack'],key,action,pc['refs'])
        cp=joblib.load(ROOT/ctx['refs'][key,action]['checkpoint'])
        x=run.parent.parent.features(raw,d,scale,cp['identity']['cutoff'])
        assert run.base.array_hash(ids,x,d)==r['source_inputs_sha256']
        assert run.base.array_hash(q)==r['source_targets_sha256']
        h=data['geometry'][ids,:16].reshape(-1,8,2)
        use=(draws>0)&pr['known']&(d>0)&np.any(h[:,-1]!=h[:,-2],axis=1)
        event=(q[:,5]==1)&(q[:,4]==0)&pr['known']
        np.testing.assert_array_equal(cp['sample_weight'],draws*(d>0))
        z=run.standardized(x,cp['preprocess'])[use].astype(float)
        values,p,_,cutoff=run.base.causal_view(pc['pack'][1],key,action)
        held=values['ids']; raw,dd,_=run.base.native_features(data['geometry'][held],p,data['scale'][held])
        xx=run.parent.parent.features(raw,dd,data['scale'][held],cutoff)
        zz=run.standardized(xx,cp['preprocess']).astype(float)
        for case in cases:
            pos=np.flatnonzero(held==case['row']); assert len(pos)==1
            source=z[:,hc] if case['arm']=='history' else z
            query=zz[pos][:,hc] if case['arm']=='history' else zz[pos]
            dist=cdist(query,source,metric='sqeuclidean')[0]/source.shape[1]
            order=np.lexsort((ids[use],dist)); near=float(np.sqrt(dist[order[0]]))
            np.testing.assert_allclose(near,case['nearest_distance'],rtol=1e-12,atol=1e-14)
            max_delta=max(max_delta,abs(near-case['nearest_distance']))
            e=event[use]; first=int(np.flatnonzero(e[order])[0]+1)
            assert first==case['nearest_zero_rank']
            exact=dist==0; assert int(exact.sum())==case['exact_rows']
            assert int((exact&e).sum())==case['exact_zero_rows']
            for k in cfg['neighbor_counts']:
                ii=order[:k]; old=case['neighborhoods'][str(k)]
                assert int(e[ii].sum())==old['zero_rows']
                assert len(np.unique(data['tracks'][ids[use][ii]]))==old['tracks']
                b,harm=q[use,0][ii],q[use,1][ii]
                assert int((b>harm).sum())==old['beneficial_rows']
                assert int((harm>b).sum())==old['harmful_rows']
            checked+=1
        print(json.dumps(dict(view=key,action=action,case_feature_comparisons=checked)),flush=True)
    assert files==36 and checked==126
    run.base.assert_current(identity)
    report=dict(result_source='fresh_run',method='scipy_cdist_separate_arithmetic_same_executor',
        view_hashes_checked=files,case_feature_comparisons=checked,max_nearest_distance_difference=max_delta,
        exact_ranks_labels_tracks=True,analysis_sha256=run.base.file_digest(public/'analysis.json'),
        code_sha256=run.base.file_digest(Path(__file__)),independent_confirmation=False)
    run.base.immutable_json(public/'separate_checks.json',report); print(json.dumps(report))


if __name__=='__main__': main()
