"""Check whether feature extrapolation survives the frozen causal gate."""
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


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, data, views, _, heads, predictions, identity = load()
    public = ROOT/cfg['reports']; path = public/'analysis.json'
    report = json.loads(path.read_text()); assert report['identity'] == identity
    records = []
    for key, meta in views.items():
        with np.load(ROOT/predictions[key]['path'], allow_pickle=False) as z:
            ids, p = z['ids'].copy(), z['prediction'].copy()
        x, d, _ = features(data['geometry'][ids], p, data['scale'][ids])
        cp = torch.load(ROOT/heads[key, 'direct_native']['checkpoint'], map_location='cpu', weights_only=False)
        z = np.abs((x.astype(float)-cp['preprocess']['mean'])/cp['preprocess']['std'])
        rec = next(r for r in report['archives'] if r['view'] == key)
        assert file_digest(ROOT/rec['path']) == rec['sha256']
        with np.load(ROOT/rec['path'], allow_pickle=False) as a:
            np.testing.assert_array_equal(ids, a['ids'])
            pools = {k:a[k].copy() for k in ('past_stop', 'bounded_fraction_strict_stop')}
        pools['all'] = np.ones(len(ids), bool)
        pools['stopped_or_same'] = ~pools['past_stop']
        history = data['geometry'][ids, :16].reshape(-1, 8, 2)
        np.testing.assert_array_equal(pools['past_stop'], (d > 0) & np.any(history[:, -1] != history[:, -2], axis=1))
        slices = {}
        for name, use in pools.items():
            slices[name] = dict(rows=int(use.sum()),
                feature_abs_z_gt10_fraction=None if not use.any() else float((z[use] > 10).mean()),
                first_candidate_lateral_gt10_fraction=None if not use.any() else float((z[use, 331] > 10).mean()),
                first_candidate_lateral_max_abs_z=None if not use.any() else float(z[use, 331].max()),
                first_candidate_lateral_native_abs_p99=None if not use.any() else float(np.quantile(
                    np.abs(p[use, 0, 1])*data['scale'][ids[use]], .99)))
        records.append(dict(view=key, slices=slices))
    result = dict(analysis_sha256=file_digest(path), result_source='fresh_run_post_readout_causal_support_diagnosis',
        auditor_sha256=file_digest(Path(__file__)), views=records, outcomes_loaded=False,
        thresholds_changed=False, decisions_changed=False, new_training=False,
        interpretation='extrapolation_association_not_causal_attribution', deployment=False)
    assert_current(identity)
    immutable_json(public/'support_forensics.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'views'}, indent=2))


if __name__ == '__main__': main()
