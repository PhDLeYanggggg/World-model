"""Count observed-input support for unit equivariance, without reading labels."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_dimensionless_refit as run
import numpy as np
import torch
from src.world_model.m3w_native_forecast import pack_geometry
from src.world_model.m3w_partial_context import condition_partial_inputs


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reg=json.loads((run.PUBLIC/'registration.json').read_text())
    for p,h in reg['bindings'].items():
        assert run.digest(ROOT/p)==h
    ref=reg['geometry']
    assert run.artifact(ROOT/ref['path'])==ref
    geo=json.loads((ROOT/ref['path']).read_text())['geometry']
    assert run.artifact(ROOT/geo['path'])==geo
    g=np.load(ROOT/geo['path'],mmap_mode='r',allow_pickle=False)
    extents=[]
    for start in range(0,len(g),8192):
        x=pack_geometry(g[start:start+8192])
        h,n,m=x['history'],x['neighbors'],x['neighbor_mask']
        safe=torch.where(m[...,None],n,0.)
        raw=torch.maximum(h[...,:2].norm(dim=-1).amax(1),safe[...,:2].norm(dim=-1).flatten(1).amax(1))
        _,scale=condition_partial_inputs(x)
        torch.testing.assert_close(raw.clamp_min(1.),scale,rtol=0,atol=0)
        extents.append(raw.numpy())
    extent=np.concatenate(extents)
    assert len(extent)==318969 and np.isfinite(extent).all()
    doc=dict(result_source='fresh_observed_geometry_only',geometry=geo,rows=len(extent),
        quantile_levels=[0,.01,.1,.5,.9,.99,1],
        observed_extent_quantiles=np.quantile(extent,[0,.01,.1,.5,.9,.99,1]).tolist(),
        extent_below_one=int((extent<1).sum()),extent_equal_one=int((extent==1).sum()),
        extent_at_least_four=int((extent>=4).sum()),zero_extent=int((extent==0).sum()),
        raw_future_labels_read=False,predictive_accuracy_measured=False,independent_roles_read=False)
    run.immutable_json(run.PUBLIC/'scale_support.json',doc)
    lines=['# Observed Scale Support','',
        f"All {len(extent):,} source query histories were checked; no future labels or forecast errors were read.",
        'Extent is the maximum past ego/valid-neighbor displacement from the current ego, in image-local units.',
        'The inherited conditioning scale is max(extent,1). This is not a verified physical scale.','',
        f"- Extent below1 (clamp strictly active): {doc['extent_below_one']:,}.",
        f"- Extent exactly1: {doc['extent_equal_one']:,}.",
        f"- Extent at least4 (supports all0.25/0.5/2/4 probes): {doc['extent_at_least_four']:,}.",
        f"- Zero extent: {doc['zero_extent']:,}.",
        f"- Quantiles {doc['quantile_levels']}: {doc['observed_extent_quantiles']}.",'',
        'A successful equivariance check on unclamped histories does not cover arbitrary unit changes',
        'that cross the clamp, nor prove predictive generalization. No rows were filtered from training or scoring.']
    (run.PUBLIC/'scale_support.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:doc[k] for k in ('rows','extent_below_one','extent_at_least_four')}))


if __name__=='__main__':
    main()
