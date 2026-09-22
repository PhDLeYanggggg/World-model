"""Post-readout fixed veto diagnosis; no training, selection or threshold change."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_prefix_cost import load, load_states, training_arrays
from scripts.run_m3w_bounded_cost import read_arrays
from scripts.run_m3w_native_forecast import immutable_json, assert_current
from src.world_model.m3w_prefix_cost_head import build, predict
from src.world_model.m3w_prefix_cost_targets import supervised_prefix_costs
from src.evaluation.m3w_conditional_cost_audit import strict_bits
from src.evaluation.m3w_prefix_veto_audit import diagnose
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, data, views, predictions, scalar, refs, identity = load()
    records, states = load_states(cfg, views, identity)
    public = ROOT / cfg['reports']; report = json.loads((public / 'analysis.json').read_text())
    assert report['identity'] == identity
    rows = []
    for key, meta in views.items():
        ids, x, costs, d, pr, _, _ = training_arrays(meta, data, refs, key, cfg)
        model = build(x.shape[1], cfg['training']['width'], meta['seed'])
        model.load_state_dict(states[key, 'prefix']['model'])
        score = predict(model, x, d, pr)
        nominal = strict_bits(score[:, -1], data['geometry'][ids, :16].reshape(-1, 8, 2), d[:, -1])
        guarded = nominal & np.all(score[..., 1] <= .1*score[..., 0], axis=1)
        valid = np.logical_and.accumulate(read_arrays(data, ids, 'valid'), 1)
        for site in sorted(set(data['sites'][ids])):
            use = data['sites'][ids] == site
            rows.append(dict(view=key, role='fitting', site=str(site),
                             **diagnose(score[use], costs[use], valid[use], nominal[use], guarded[use])))
        archive = next(r for r in report['archives'] if r['view'] == key)
        assert file_digest(ROOT / archive['path']) == archive['sha256']
        with np.load(ROOT / archive['path'], allow_pickle=False) as z:
            hi, hs, hn, hg = z['ids'].copy(), z['profile'].copy(), z['profile_terminal'].copy(), z['profile_guard'].copy()
        with np.load(ROOT / predictions[key]['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(hi, z['ids']); hp = z['prediction'].copy()
        labels = supervised_prefix_costs(data['geometry'][hi, 332:356].reshape(-1, 12, 2), hp,
                    read_arrays(data, hi, 'target'), read_arrays(data, hi, 'valid'), data['scale'][hi])
        rows.append(dict(view=key, role='held_source', site=meta['outer_site'],
                         **diagnose(hs, labels['costs'], labels['available'], hn, hg)))
    assert_current(identity)
    out = dict(result_source='fresh_post_readout_frozen_veto_diagnosis',
               analysis_sha256=file_digest(public / 'analysis.json'), code_sha256=file_digest(Path(__file__)),
               helper_sha256=file_digest(ROOT / 'src/evaluation/m3w_prefix_veto_audit.py'),
               tests_sha256=file_digest(ROOT / 'tests/test_m3w_prefix_veto_audit.py'),
               records=rows, new_training=False, changed_decisions=False, independent_confirmation=False)
    immutable_json(public / 'veto_diagnosis.json', out)
    print(json.dumps(dict(records=len(rows), source=out['result_source'], changed_decisions=False)))


if __name__ == '__main__':
    main()
