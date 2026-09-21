"""Separate native-error arithmetic and matched fit support verification."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.run_m3w_native_eqmotion import load, trial_identity
from scripts.run_m3w_native_forecast import assert_current, immutable_json
from scripts.verify_m3w_native_joint_controls import distances, check_reduction
from src.evaluation.m3w_experiment_contract import file_digest


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cfg, data, rows, views, identity = load()
    public = ROOT/cfg['reports']
    report = json.loads((public/'analysis.json').read_text())
    replay = json.loads((public/'replay.json').read_text())
    assert report['identity'] == identity and replay['all_checks_passed']
    assert replay['analysis_sha256'] == file_digest(public/'analysis.json')
    n = len(data['sites'])
    cv, cf = distances(data['geometry'][:, 332:356].reshape(n, 12, 2),
        data['target'], data['valid'], data['scale'])
    pos = np.broadcast_to(data['geometry'][:, 14:16, None].transpose(0, 2, 1), (n, 12, 2))
    ca, ce = distances(pos, data['target'], data['valid'], data['scale'])
    masks = dict(complete=data['valid'].all(1), static_history=rows['static_history'],
        moving_history=~rows['static_history'], hard=np.zeros(n, bool), positive_easy=np.zeros(n, bool),
        zero_CV=data['valid'].all(1) & (cv == 0))
    errors = {name:{s:[np.full(n, np.nan), np.full(n, np.nan)] for s in cfg['seeds']}
              for name in ('eqmotion', 'transformer')}
    seen = {s:np.zeros(n, bool) for s in cfg['seeds']}
    fitted_rows = 0
    for key, view in views.items():
        record = next(r for r in report['training'] if r['view'] == key)
        assert file_digest(ROOT/record['checkpoint']) == record['checkpoint_sha256']
        cp = torch.load(ROOT/record['checkpoint'], map_location='cpu', weights_only=False)
        ref = torch.load(ROOT/view['reference']['checkpoint'], map_location='cpu', weights_only=False)
        assert cp['identity'] == trial_identity(identity, key, view)
        fold, seed = view['fold'], view['seed']
        np.testing.assert_array_equal(cp['train_ids'], np.flatnonzero(data['sites'] != view['site']))
        assert cp['step'] == ref['step'] == cfg['training']['steps']
        for field in ('train_ids', 'draws', 'factors'):
            np.testing.assert_array_equal(cp[field], ref[field])
        assert cp['draws'].sum() == cfg['training']['steps']*cfg['training']['batch_size']
        assert cp['draws'][data['sites'] == view['site']].sum() == 0
        for site in set(data['sites'])-{view['site']}:
            use = data['sites'] == site
            supported = use & np.isfinite(cv)
            expected = data['scale'][use]*use.sum()/supported.sum()/cv[supported].mean()
            np.testing.assert_allclose(cp['factors'][use], expected, rtol=1e-12, atol=1e-10)
        fitted_rows += len(cp['train_ids'])
        ids = fold['held_ids']
        assert not seen[seed][ids].any()
        seen[seed][ids] = True
        paths = dict(eqmotion=next(r for r in report['archives'] if r['view'] == key),
                     transformer=view['reference_prediction'])
        for name, artifact in paths.items():
            assert file_digest(ROOT/artifact['path']) == artifact['sha256']
            with np.load(ROOT/artifact['path'], allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], ids)
                ade, fde = distances(z['prediction'], data['target'][ids], data['valid'][ids], data['scale'][ids])
            errors[name][seed][0][ids], errors[name][seed][1][ids] = ade, fde
        train = cp['train_ids']
        supported = cv[train][np.isfinite(cv[train])]
        masks['hard'][ids] = cv[ids] >= np.quantile(supported, .75)
        masks['positive_easy'][ids] = (cv[ids] > 0) & (cv[ids] <= np.quantile(supported[supported > 0], .25))
    assert all(x.all() for x in seen.values())
    def verify_summary(ade, fde, result):
        reductions = check_reduction(ade, cv, data['sites'], cfg['sites'], result['ADE'])
        reductions += check_reduction(fde, cf, data['sites'], cfg['sites'], result['FDE'])
        assert int((ade[masks['zero_CV']] > 0).sum()) == result['zero_CV_harmed']
        for group, mask in masks.items():
            reductions += check_reduction(ade[mask], cv[mask], data['sites'][mask], cfg['sites'], result['subsets'][group])
        return reductions
    reductions = 0
    for name, group in errors.items():
        for seed, values in group.items():
            reductions += verify_summary(*values, report['summaries'][name]['seeds'][str(seed)])
        reductions += verify_summary(*np.mean(list(group.values()), axis=0), report['summaries'][name]['mean_seed'])
    reductions += verify_summary(cv, cf, report['summaries']['constant_velocity'])
    reductions += verify_summary(ca, ce, report['summaries']['constant_position'])
    left, right = (report['summaries'][k]['mean_seed']['ADE']['by_scene'] for k in ('eqmotion', 'transformer'))
    difference = np.array([left[s]['gain_percent']-right[s]['gain_percent'] for s in cfg['sites']])
    r = report['primary_contrast']
    np.testing.assert_allclose(difference, r['scene_differences_pp'], atol=1e-10, rtol=1e-10)
    rng = np.random.default_rng(r['seed'])
    indices = rng.integers(len(difference), size=(r['resamples'], len(difference)))
    np.testing.assert_allclose(np.quantile(difference[indices].mean(1), [.025, .975]), r['ci95_pp'], atol=1e-10)
    result = dict(analysis_sha256=file_digest(public/'analysis.json'), all_checks_passed=True,
        verifier_sha256=file_digest(Path(__file__)),
        reduction_helper_sha256=file_digest(ROOT/'scripts/verify_m3w_native_joint_controls.py'),
        matched_fits=len(views), training_rows_checked_repeated=fitted_rows,
        scene_reductions=reductions, independent_implementation_same_agent=True,
        independent_research_confirmation=False, selection_changes=False, deployment=False)
    assert_current(identity)
    immutable_json(public/'independent_verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
