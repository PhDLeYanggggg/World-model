"""Descriptive frozen-cache quality,never downstream normalization or fitting."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import run_m3w_eqmotion_context_cache_v2 as cache


def main():
    freeze = json.loads((cache.PUBLIC/'cache_freeze.json').read_text())
    rows = []
    for ref in freeze['caches']:
        meta = json.loads(cache.checked(ref).read_text())
        x = np.load(cache.checked(meta['latent']),mmap_mode='r',allow_pickle=False)
        ids = np.load(cache.checked(meta['ids']),allow_pickle=False)
        if len(ids)!=len(x) or x.shape[1:]!=(64,) or np.any(np.diff(ids)<=0):
            raise ValueError('Frozen sorted IDs and h64shape required')
        total,second = np.zeros(64),np.zeros(64)
        low,high = float('inf'),-float('inf')
        for start in range(0,len(x),8192):
            values = x[start:start+8192].astype(float)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite frozen features')
            total += values.sum(0)
            second += np.square(values).sum(0)
            low,high = min(low,float(values.min())),max(high,float(values.max()))
        std = np.sqrt(np.maximum(second/len(x)-np.square(total/len(x)),0))
        rows.append(dict(trial=meta['identity']['trial'],rows=len(x),dimension=64,minimum=low,maximum=high,
                         std_minimum=float(std.min()),std_median=float(np.median(std)),std_maximum=float(std.max()),
                         dimensions_with_std_above_1e_minus_6=int((std>1e-6).sum()),latent=meta['latent'],ids=meta['ids']))
    if len(rows)!=9 or sum(r['rows'] for r in rows)!=897060:
        raise ValueError('All nine frozen producer caches required')
    out = dict(result_source='fresh_descriptive_cache_scan_not_training_or_feature_utility',pid=os.getpid(),
               caches=rows,cache_freeze=cache.artifact(cache.PUBLIC/'cache_freeze.json'),
               diagnostic_code=cache.artifact(Path(__file__)),normalization_fitted=False,
               global_statistics_used_by_downstream_learner=False,future_labels_used=False,
               JEPA_noncollapse_claim=False,feature_lift_evaluated=False,deployment_changed=False)
    cache.write(cache.PUBLIC/'quality.json',out)
    print(json.dumps(out))


if __name__=='__main__':
    main()
