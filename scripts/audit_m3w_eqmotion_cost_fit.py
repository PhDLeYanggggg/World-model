"""Frozen head fit versus held-source audit in training-defined strata."""
import json
import os
from pathlib import Path
import platform
import sys
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 environment required before Torch import')
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.run_m3w_eqmotion_cost_refit import load
from scripts.run_m3w_bounded_cost import read_arrays, training_data, features, selections
from scripts.run_m3w_native_forecast import assert_current, immutable_json, array_hash
from scripts.verify_m3w_native_joint_controls import distances
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_cost_support_audit import fit_cuts, strata, cost_stats
from src.world_model.m3w_bounded_cost_head import ARMS, build, predict

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/eqmotion_cost_fit_forensics_v1'


def motion(data, ids):
    past = data['geometry'][ids, :16].reshape(-1, 8, 2)
    return past, np.linalg.norm(past[:, -1].astype(float)-past[:, -2], axis=1)*data['scale'][ids]


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, data, views, _, predictions, _, identity = load()
    prior = ROOT/cfg['reports']; report = json.loads((prior/'analysis.json').read_text())
    verified = json.loads((prior/'independent_verification.json').read_text())
    assert verified['all_checks_passed'] and verified['analysis_sha256'] == file_digest(prior/'analysis.json')
    assert report['identity'] == identity
    local_sources = [Path(__file__), ROOT/'src/evaluation/m3w_cost_support_audit.py',
                     ROOT/'tests/test_m3w_cost_support_audit.py', PUBLIC/'design.md']
    bindings = {str(p.relative_to(ROOT)):file_digest(p) for p in local_sources}
    artifact_identity = dict(parent_analysis_sha256=file_digest(prior/'analysis.json'), source_bindings=bindings)
    immutable_json(PUBLIC/'identity.json', artifact_identity)
    rows, cuts, traces = [], {}, []
    for key, meta in views.items():
        ids, x, y, d, pr = training_data(meta, data)
        past, speed = motion(data, ids)
        cuts[key] = fit_cuts(d, speed, pr['known'])
        weights = pr['weights']; den = np.where(d > 0, d, 1.)
        fractions = np.where(pr['known'][:, None], y/den[:, None], 0)
        constant_fraction = (weights[:, None]*fractions).sum(0)
        assert (constant_fraction >= 0).all() and constant_fraction.sum() <= 1+1e-6
        with np.load(ROOT/predictions[key]['path'], allow_pickle=False) as z:
            held_ids, p = z['ids'].copy(), z['prediction'].copy()
        np.testing.assert_array_equal(held_ids, np.flatnonzero(data['sites'] == meta['outer_site']))
        hx, hd, _ = features(data['geometry'][held_ids], p, data['scale'][held_ids])
        hpast, hspeed = motion(data, held_ids)
        target, valid = read_arrays(data, held_ids, 'target'), read_arrays(data, held_ids, 'valid')
        base = data['geometry'][held_ids, 332:356].reshape(-1, 12, 2)
        cv, _ = distances(base, target, valid, data['scale'][held_ids])
        error, _ = distances(p, target, valid, data['scale'][held_ids])
        hy = np.column_stack((np.maximum(cv-error, 0), np.maximum(error-cv, 0)))
        hy[~valid.all(1)] = np.nan
        archive = next(r for r in report['archives'] if r['view'] == key)
        assert file_digest(ROOT/archive['path']) == archive['sha256']
        for arm in ARMS:
            r = next(r for r in report['training'] if r['view'] == key and r['arm'] == arm)
            assert file_digest(ROOT/r['checkpoint']) == r['checkpoint_sha256']
            cp = torch.load(ROOT/r['checkpoint'], map_location='cpu', weights_only=False)
            assert cp['identity']['inputs_sha256'] == array_hash(ids, x, d)
            assert cp['identity']['labels_sha256'] == array_hash(y)
            model = build(x.shape[1], cfg['training']['width'], meta['seed']); model.load_state_dict(cp['model'])
            train_score, held_score = predict(model, x, d, pr, arm), predict(model, hx, hd, pr, arm)
            with np.load(ROOT/archive['path'], allow_pickle=False) as z:
                np.testing.assert_array_equal(held_score, z['refit_'+arm])
            traces.append(dict(view=key, arm=arm, first_loss=cp['trace'][0]['loss'],
                last_loss=cp['trace'][-1]['loss'], final_five_loss_mean=float(np.mean([r['loss'] for r in cp['trace'][-5:]])),
                loss_trace_is_minibatch_not_validation=True))
            for population, qids, score, labels, disagreement, movement, history, support in (
                ('fitting', ids, train_score, y, d, speed, past, pr['known']),
                ('held_source', held_ids, held_score, hy, hd, hspeed, hpast, valid.all(1))):
                sets = {'all':np.ones(len(qids), bool)}
                sets.update({'disagreement_'+k:v for k,v in strata(disagreement,cuts[key]['disagreement']).items()})
                sets.update({'speed_'+k:v for k,v in strata(movement,cuts[key]['past_step_displacement']).items()})
                bits = selections(score, history, disagreement, np.zeros(len(qids), bool), qids)
                sets.update({k:bits[k] for k in ('net_stop', 'strict_stop')})
                sets.update({'site_'+s:data['sites'][qids] == s for s in sorted(set(data['sites'][qids]))})
                for label, mask in sets.items():
                    rows.append(dict(view=key, arm=arm, population=population, stratum=label,
                        **cost_stats(score, labels, disagreement, mask, support, pr['constant'], constant_fraction)))
        print(json.dumps(dict(state='view_complete', view=key, fitting_rows=len(ids), held_rows=len(held_ids))), flush=True)
    out = dict(identity=artifact_identity, result_source='fresh_frozen_checkpoint_diagnostic_no_training',
        parent_analysis_sha256=file_digest(prior/'analysis.json'), cuts=cuts, rows=rows, traces=traces,
        checkpoint_views=36, held_score_rows_replayed=175756*3*3,
        fit_is_in_sample_optimistic=True, held_sources_design_exposed=True,
        count_is_repeated_views_not_independent_people=True,
        threshold_search=False, new_training=False, choice_changes=False,
        independent_confirmation=False, deployment=False, stage5c_executed=False, smc_enabled=False)
    assert_current(identity); assert_current(artifact_identity); immutable_json(PUBLIC/'analysis.json', out)
    print(json.dumps(dict(state='completed', rows=len(rows), choices_changed=False,
                         analysis_sha256=file_digest(PUBLIC/'analysis.json'))))


if __name__ == '__main__': main()
