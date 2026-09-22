"""Fixed fit/held conditional-risk diagnostic; never selects a forest or threshold."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_forest_cost import load, load_states
from scripts.run_m3w_temporal_intervention import training_arrays
from scripts.run_m3w_bounded_cost import read_arrays
from scripts.run_m3w_native_forecast import assert_current, immutable_json
from scripts.verify_m3w_temporal_intervention import manual_candidates
from scripts.verify_m3w_native_joint_controls import distances
from src.world_model.m3w_forest_cost_head import predict, ARMS
from src.world_model.m3w_bounded_cost_head import build, predict as neural_predict
from src.evaluation.m3w_conditional_cost_audit import strict_bits
from src.evaluation.m3w_temporal_fit_support import population_summary
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, old, data, views, predictions, refs, previous, prior, identity = load()
    _, states = load_states(cfg, views, identity); public = ROOT/cfg['reports']
    report = json.loads((public/'analysis.json').read_text()); digest = file_digest(public/'analysis.json')
    assert report['identity'] == identity
    for name in ('replay.json', 'separate_verification.json'):
        r = json.loads((public/name).read_text()); assert r['all_checks_passed'] and r['analysis_sha256'] == digest
    n = len(data['sites']); past = data['geometry'][:, :16].reshape(n, 8, 2)
    target, valid = read_arrays(data, np.arange(n), 'target'), read_arrays(data, np.arange(n), 'valid')
    baseline = data['geometry'][:, 332:356].reshape(n, 12, 2)
    cv, _ = distances(baseline, target, valid, data['scale']); full = valid.all(1)
    records = []
    for key, meta in views.items():
        archive = next(r for r in report['archives'] if r['view'] == key)
        assert file_digest(ROOT/archive['path']) == archive['sha256']
        with np.load(ROOT/archive['path'], allow_pickle=False) as z: held = {k: z[k].copy() for k in z.files}
        hi = held['ids']
        with np.load(ROOT/predictions[key]['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(hi, z['ids']); p = z['prediction'].copy()
        hq, _ = manual_candidates(baseline[hi], p)
        for arm in ARMS:
            ids, x, y, d, pr, _, w, _ = training_arrays(meta, data, refs, key, old, arm)
            cp = states[key, arm]; forest = predict(cp['model'], x, d, pr)
            prev = previous[key, arm]
            model = build(x.shape[1], old['training']['width'], meta['seed']); model.load_state_dict(prev['model'])
            neural = neural_predict(model, x, d, pr, 'bounded_native')
            herror, _ = distances(hq[arm], target[hi], valid[hi], data['scale'][hi])
            hy = np.column_stack((np.maximum(cv[hi]-herror, 0), np.maximum(herror-cv[hi], 0)))
            hy[~full[hi]] = np.nan
            for estimator, score in (('forest', forest), ('neural', neural)):
                choice = strict_bits(score, past[ids], d)
                for site in pr['training_sites']:
                    m = data['sites'][ids] == site
                    records.append(dict(view=key, arm=arm, estimator=estimator, role='fitting', site=site,
                        costs=population_summary(y[m], score[m], d[m], choice[m], w[m])))
                records.append(dict(view=key, arm=arm, estimator=estimator, role='held_source', site=meta['outer_site'],
                    costs=population_summary(hy, held[arm+'_'+estimator+'_score'], held[arm+'_distance'],
                        held[arm+'_'+estimator], np.ones(len(hi)))))
        print(json.dumps(dict(view=key, state='fit_and_held_diagnosed')), flush=True)
    summary = {}
    for arm in ARMS:
        for estimator in ('forest', 'neural'):
            for role in ('fitting', 'held_source'):
                rows = [r['costs']['selected'] for r in records
                        if r['arm'] == arm and r['estimator'] == estimator and r['role'] == role]
                supported = [r for r in rows if r['complete']]
                ratios = [r['harm_ratio'] for r in supported if r['harm_ratio'] is not None]
                summary[arm+'_'+estimator+'_'+role] = dict(groups=len(rows), supported_groups=len(supported),
                    selected_underprediction=sum(r['harm_underpredicted'] for r in supported),
                    selected_complete_rows=sum(r['complete'] for r in rows),
                    selected_harm_ratio_quantiles=np.quantile(ratios, [0, .5, 1]).tolist() if ratios else None)
    assert len(records) == 192
    result = dict(result_source='fresh_readonly_forest_neural_fit_and_held_diagnosis', analysis_sha256=digest,
        code_sha256=file_digest(Path(__file__)), records=records, summary=summary,
        no_new_fit=True, changed_decisions=False, fitting_diagnostics_in_sample_not_calibration=True,
        independent_confirmation=False, deployment=False, stage5c_executed=False, smc_enabled=False)
    assert_current(identity); immutable_json(public/'fit_held_diagnosis.json', result)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
