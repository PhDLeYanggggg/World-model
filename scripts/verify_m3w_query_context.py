"""Independently reduce the frozen query-context readout; never alter decisions."""
import argparse
from collections import defaultdict
import json

from scripts import run_m3w_query_context_risk as run


def calculate():
    reg, cfg, _ = run.admission(); freeze, docs = run.models(reg)
    complete = json.loads((run.PUBLIC/'complete.json').read_text())
    summary = json.loads((run.PUBLIC/'summary.json').read_text())
    if complete['summary_sha256'] != run.sha(run.PUBLIC/'summary.json'): raise ValueError('Changed summary')
    rows = []
    for ref in complete['readouts']:
        p = run.ROOT/ref['path']
        if run.sha(p) != ref['sha256']: raise ValueError('Changed readout')
        rows.append(json.loads(p.read_text()))
    if {(r['context'], r['source'], r['seed']) for r in rows} != run.expected(reg):
        raise ValueError('Wrong readout roster')
    checks = 0
    for field, estimate in summary['contrasts'].items():
        grouped = defaultdict(list)
        for row in rows: grouped[row['source']].append(row['result']['contrasts'][field])
        if any(v is None for values in grouped.values() for v in values):
            assert estimate['mean'] is None and estimate['CI95'] is None
            continue
        local = [sum(grouped[s])/len(grouped[s]) for s in sorted(grouped)]
        rng = run.api.np.random.default_rng(cfg['bootstrap_seed'])
        samples = []
        for _ in range(cfg['bootstrap_draws']):
            ix = rng.integers(0, len(local), len(local))
            samples.append(sum(local[int(i)] for i in ix)/len(local))
        run.api.np.testing.assert_allclose(estimate['mean'], sum(local)/len(local), atol=1e-12)
        run.api.np.testing.assert_allclose(estimate['CI95'], run.api.np.quantile(samples, [.025, .975]), atol=1e-12)
        checks += 1
    for arm, v in summary['safety'].items():
        ps = [r['result']['policies'][arm] for r in rows]
        assert v['selected'] == sum(p['selected_count'] for p in ps)
        assert v['unknown_selected'] == sum(p['selected_unknown'] for p in ps)
        assert v['full_support'] == sum(p['finite_completion_supported'] for p in ps)
        checks += 3
    pairs, parameter_counts = [], set()
    for context in sorted(reg['contexts']):
        for seed in reg['seeds']:
            states = [run.api.core.read_checkpoint(run.ROOT/docs[context, seed, a]['checkpoint']['path']) for a in run.api.ARMS]
            run.api.assert_matched(*states)
            for k in ('loss', 'primary_loss', 'regret', 'mean_soft_action'):
                run.api.core.exact(states[0]['trace'][0][k], states[1]['trace'][0][k])
            parameter_counts.add((states[0]['trainable_parameters'], states[0]['total_parameters']))
            encoder_changes = {a: sum(float((v.double()-s['initial_model'][k].double()).square().sum())
                for k, v in s['model'].items() if k.startswith('risk_encoder.'))
                for a, s in zip(run.api.ARMS, states)}
            if min(encoder_changes.values()) <= 0 or states[0]['trainable_parameters'] != 12322:
                raise ValueError('The repaired real encoder must actually train')
            pairs.append(dict(context=context, seed=seed,
                encoder_squared_weight_changes=encoder_changes,
                final_sampled_primary_loss={a: s['trace'][-1]['primary_loss'] for a, s in zip(run.api.ARMS, states)},
                logged_gradient_clip_fraction={a: sum(r['gradient_norm'] > s['settings']['gradient_clip'] for r in s['trace'])/len(s['trace']) for a, s in zip(run.api.ARMS, states)},
                final_dual_at_cap={a: int((s['dual'] >= s['objective_config']['dual_maximum']).sum()) for a, s in zip(run.api.ARMS, states)}))
    availability = {r['context']+'_'+r['source']: r['context_stats'] for r in rows if r['seed'] == 17}
    per_seed = {}
    for seed in reg['seeds']:
        subset = [r for r in rows if r['seed'] == seed]
        per_seed[str(seed)] = {}
        for field in ('self_full', 'self_matched', 'rational_full', 'rational_matched'):
            groups = defaultdict(list)
            for r in subset: groups[r['source']].append(r['result']['contrasts'][field])
            per_seed[str(seed)][field] = sum(sum(v)/len(v) for v in groups.values())/len(groups)
    return dict(result_source='fresh_run_independent_arithmetic_and_checkpoint_verification',
        checks=checks, paired_training_checks=len(pairs), parameters=sorted(parameter_counts),
        training_pairs=pairs, held_context_availability=availability, per_seed_utility_pp=per_seed,
        optimizer_steps=sum(d['steps'] for d in docs.values()),
        summary_sha256=run.sha(run.PUBLIC/'summary.json'), freeze_sha256=run.sha(run.PUBLIC/'training_freeze.json'),
        completed_fits=freeze['fresh_fits'], independent_roles_read=False, deployment_changed=False,
        limit='Same-agent verification; exploratory, not independent research replication')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--check', action='store_true'); a = p.parse_args()
    doc = calculate(); path = run.PUBLIC/'verification.json'
    if a.check:
        if json.loads(path.read_text()) != doc: raise ValueError('Verification changed')
    else: run.once(path, doc)
    print(json.dumps({k: v for k, v in doc.items() if k not in ('training_pairs', 'held_context_availability')}))
