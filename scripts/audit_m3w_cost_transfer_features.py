"""Post-readout, past-only feature shift diagnosis; does not alter any decision."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.run_m3w_cost_head_transfer import load, features
from scripts.run_m3w_native_forecast import immutable_json, assert_current
from src.evaluation.m3w_experiment_contract import file_digest


def feature_names():
    names = [f'history_{t}_{a}' for t in range(8) for a in ('x', 'y', 'time')]
    names += [f'history_mask_{t}' for t in range(8)]
    names += [f'neighbor_{i}_history_{t}_{a}' for i in range(8) for t in range(8) for a in ('x', 'y', 'time')]
    names += [f'neighbor_{i}_mask_{t}' for i in range(8) for t in range(8)]
    names += [f'{p}_{stat}_{a}' for p in ('baseline', 'candidate', 'disagreement')
              for stat in ('mean', 'std', 'endpoint') for a in ('x', 'y')]
    names += [f'{p}_rollout_{t}_{a}' for p in ('baseline', 'candidate') for t in range(12) for a in ('x', 'y')]
    names += ['log_past_scale', 'log1p_native_disagreement']
    assert len(names) == 356
    return names


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, data, views, _, heads, predictions, identity = load()
    public = ROOT/cfg['reports']; analysis = public/'analysis.json'
    assert json.loads(analysis.read_text())['identity'] == identity
    names, records = feature_names(), []
    candidate_columns = np.r_[294:306, 330:354, 355]
    for key, meta in views.items():
        with np.load(ROOT/predictions[key]['path'], allow_pickle=False) as z:
            ids, pe = z['ids'].copy(), z['prediction'].copy()
        with np.load(ROOT/meta['outer_prediction']['prediction']['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids']); pt = z['prediction'].copy()
        xe, _, _ = features(data['geometry'][ids], pe, data['scale'][ids])
        xt, _, _ = features(data['geometry'][ids], pt, data['scale'][ids])
        cp = torch.load(ROOT/heads[key, 'direct_native']['checkpoint'], map_location='cpu', weights_only=False)
        pr = cp['preprocess']
        changed = np.max(np.abs(xe.astype(float)-xt.astype(float)), axis=0) > 0
        assert set(np.flatnonzero(changed)) <= set(candidate_columns)
        ze, zt = [np.abs((x.astype(float)-pr['mean'])/pr['std']) for x in (xe, xt)]
        emax, tmax = ze.max(0), zt.max(0)
        floor = pr['std'] <= 1.000001e-6
        top = np.argsort(-emax, kind='stable')[:8]
        top_candidate = candidate_columns[np.argsort(-emax[candidate_columns], kind='stable')[:6]]
        def columns(indices):
            return [dict(index=int(i), feature=names[i], fit_std=float(pr['std'][i]),
                fit_mean=float(pr['mean'][i]), candidate_dependent=bool(i in candidate_columns),
                eqmotion_max_abs_z=float(emax[i]), transformer_max_abs_z=float(tmax[i]),
                eqmotion_abs_z_gt10_fraction=float((ze[:, i] > 10).mean()),
                transformer_abs_z_gt10_fraction=float((zt[:, i] > 10).mean())) for i in indices]
        records.append(dict(view=key, rows=len(ids), fit_std_floor_columns=int(floor.sum()),
            changed_floor_columns=[names[i] for i in np.flatnonzero(floor & changed)],
            eqmotion_abs_z_gt10_fraction=float((ze > 10).mean()),
            transformer_abs_z_gt10_fraction=float((zt > 10).mean()),
            top_columns=columns(top), top_candidate_columns=columns(top_candidate)))
    result = dict(result_source='fresh_run_post_readout_past_only_feature_diagnosis',
        analysis_sha256=file_digest(analysis), auditor_sha256=file_digest(Path(__file__)),
        feature_names=names, views=records, outcomes_loaded=False, decisions_changed=False,
        normalization_refitted=False, new_training=False, causal_feature_alignment_pass=True,
        independent_confirmation=False, deployment=False)
    assert_current(identity)
    immutable_json(public/'feature_forensics.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('views', 'feature_names')}, indent=2))


if __name__ == '__main__': main()
