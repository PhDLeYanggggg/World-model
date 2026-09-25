"""Diagnose frozen score-head feature support and saturation on opened sources."""
import json
import os
from pathlib import Path
import platform
import sys

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_european_nested_calibration as old
from scripts.run_m3w_native_forecast import immutable_json, array_hash
from src.world_model.m3w_native_gain_harm import standardized, build_head

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_score_scaling_v1'


def sampled(ids, sites):
    return np.concatenate([group[np.linspace(0, len(group)-1, min(512,len(group)),dtype=int)]
        for site in sorted(set(sites[ids])) for group in [ids[sites[ids]==site]]])


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reg, _, data, identity = old.load()
    transport = ROOT/'outputs/publication_readiness_2026_09/european_producer_transport_v1'
    verify = json.loads((transport/'verification.json').read_text())
    if not verify['all_passed'] or verify['analysis_sha256'] != old.digest(transport/'analysis.json'):
        raise ValueError('Verified starting diagnosis required')
    heads = old.all_heads(reg, data, identity)
    rows = {}
    for name, candidate, fold, seed, design in old.jobs(reg, data, identity):
        a = old.assemble(candidate, fold, seed, design, data, identity)
        slices = dict(fitting=sampled(design['train_ids'], data['sites']),
            excluded=sampled(design['held_ids'], data['sites']))
        for task in ('utility', 'all', 'easy'):
            r = heads[name+'_'+task]
            state = torch.load(ROOT/r['artifacts']['checkpoint']['path'], map_location='cpu', weights_only=False)
            pr = state['preprocess']
            model = build_head(reg['head_training']['width'], pr, seed)
            model.load_state_dict(state['model']);model.eval()
            result = dict(task=task, checkpoint=r['artifacts']['checkpoint'],
                tiny_std_features=np.flatnonzero(pr['std'] <= 1.01e-6).tolist(), slices={})
            for subset, ids in slices.items():
                x = a['x'][ids]
                z = standardized(x, pr)
                with torch.no_grad():
                    logits = model.network[:-1](torch.from_numpy(z)).numpy()
                maxima = np.abs(z).max(0)
                top = np.argsort(-maxima)[:12]
                result['slices'][subset] = dict(rows=len(ids), ids_sha256=array_hash(ids),
                    row_max_abs_z_quantiles=np.quantile(np.abs(z).max(1),[0,.5,.9,.99,1]).tolist(),
                    row_fraction_any_abs_z_above={str(t):float((np.abs(z).max(1)>t).mean()) for t in (5,10,100,1000)},
                    tiny_std_changed_columns=[int(i) for i in result['tiny_std_features'] if maxima[i]>1],
                    logit_quantiles=np.quantile(logits,[0,.5,.9,.99,1],axis=0).tolist(),
                    saturation_below_minus10=(logits < -10).mean(0).tolist(),
                    saturation_below_minus20=(logits < -20).mean(0).tolist(),
                    top_features=[dict(column=int(i),max_abs_z=float(maxima[i]),std=float(pr['std'][i]),
                        mean=float(pr['mean'][i])) for i in top])
            rows[name+'_'+task] = result
        print(json.dumps(dict(pid=os.getpid(),completed=name)),flush=True)
    old.assert_identity(identity)
    immutable_json(PUBLIC/'diagnosis.json', dict(result_source='fresh_run_frozen_head_diagnosis',
        source_identity=identity, code_sha256=old.digest(Path(__file__)),
        prior_analysis_sha256=verify['analysis_sha256'], rows=rows,
        sample_rule='up_to512_evenly_spaced_per_fitting_or_excluded_locality_not_independent_replicates',
        new_training=False,reserved_roles_opened=False,deployment_changed=False))


if __name__ == '__main__':
    main()
