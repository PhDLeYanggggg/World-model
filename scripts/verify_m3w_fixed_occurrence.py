"""Verify paired training, then independent action/cost accounting and report it."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_fixed_occurrence_policy as run
from scripts.report_m3w_easy_component_diagnostic import check_node
from scripts.verify_m3w_easy_risk_priority import check_actions, check_contrasts
from scripts.verify_m3w_easy_hurdle import check_quality
from scripts.verify_m3w_query_excess_refit import check_selected_costs
from scripts.verify_m3w_subset_excess import check_fixed_denominator
from scripts.verify_m3w_fixed_floor_tail import reduce_check
from scripts.report_m3w_easy_risk_priority import BOUNDARIES, value

TESTS = ['tests/test_m3w_fixed_occurrence.py', 'tests/test_m3w_fixed_occurrence_policy.py',
    'tests/test_m3w_fixed_occurrence_verification.py',
    'tests/test_m3w_fitting_gain_labels.py', 'tests/test_m3w_easy_component_diagnostic.py',
    'tests/test_m3w_easy_component_report.py', 'tests/test_m3w_easy_hurdle_verification.py',
    'tests/test_m3w_query_excess_verification.py', 'tests/test_m3w_easy_hurdle_accounting.py']


def put(name, text):
    path = run.PUBLIC/name
    if path.exists(): assert path.read_text() == text
    else: path.write_text(text)


def verdict_for(ci, screen_pass):
    if screen_pass:
        assert ci is not None and ci[0] > 0
        return 'exploratory_screen_pass_only'
    if ci is None: return 'undefined_primary'
    if ci[0] > 0: return 'advantage_but_screen_failed'
    if ci[1] < 0: return 'negative_primary'
    return 'no_resolved_primary_advantage'


def check_record_sets(details):
    expected = dict(rows=set(run.POLICIES), qualities=set(run.api.ARMS),
                    contrasts={a+'_vs_'+b for a, b in run.CONTRASTS})
    views = None
    for key, policies in expected.items():
        records = details[key]
        triples = {(r['group'], r['site'], r['policy']) for r in records}
        assert len(records) == len(triples) == 216*len(policies)
        pairs = {(g, s) for g, s, _ in triples}
        assert len(pairs) == 216
        assert triples == {(g, s, p) for g, s in pairs for p in policies}
        if views is not None: assert pairs == views
        views = pairs


def training():
    cfg, reg = run.train.registration()
    assert reg == json.loads((run.PUBLIC/'registration.json').read_text())
    done = json.loads((run.PUBLIC/'training_freeze.json').read_text())
    replay = json.loads((run.PUBLIC/'fit_replay.json').read_text())
    assert done['heads'] == 216 and len(done['groups']) == 108 and done['parameter_updates'] == 432000
    assert len({r['path'] for r in done['groups']}) == 108
    assert done['registration_sha256'] == run.train.digest(run.PUBLIC/'registration.json')
    assert not done['held_outcomes_used'] and not done['independent_roles_read']
    assert replay['exact_except_elapsed'] and not replay['all108_pairs_retrained']
    loss = {a: [] for a in run.api.ARMS}; screens = {a: [] for a in run.api.ARMS}; improved = checks = 0
    components = {a: [] for a in run.api.ARMS}
    for ref in done['groups']:
        doc = run.checked(ref); states = {}
        assert set(doc['artifacts']) == set(run.api.ARMS)
        assert doc['identity']['registration_sha256'] == done['registration_sha256']
        for arm, item in doc['artifacts'].items():
            assert run.base.artifact(ROOT/item['path']) == item
            s = run.old.api.head.read_checkpoint(ROOT/item['path']); states[arm] = s
            assert s['identity'] == doc['identity'] and s['step'] == 2000 and not s['unknown_rows_sampled']
            for k, v in s['warm_start']['model'].items():
                for prefix in ('occurrence.', 'cost.'):
                    run.old.api.sampling.exact(v, s['initial_model'][prefix+k])
                if arm == 'fixed': run.old.api.sampling.exact(v, s['model']['occurrence.'+k])
        run.api.assert_matched(states['trainable'], states['fixed'])
        assert doc['frozen_occurrence_predictions_exact'] and not doc['held_outcomes_used']
        account = doc['fitting_accounting']; keys = dict(trainable='uncapped', fixed='risk_priority')
        roles = doc['identity']['source_identity']['roles']
        assert set(account['sources']) == set(roles['training_sites'])
        for part in [account['source_balanced'], *account['sources'].values()]: checks += check_node(part)
        for arm, k in keys.items():
            part = account['source_balanced']['arms'][k]
            loss[arm].append(part['objective']['marginal'])
            components[arm].append({**part['objective'], **{key: part[key] for key in
                ('Brier', 'signed_bias', 'conditional_reference_MSE', 'conditional_harm_MSE')}})
            screens[arm].append(doc['fitting_sign_screen'][arm])
        improved += loss['fixed'][-1] < loss['trainable'][-1]
    screen = {}
    for arm, rows in screens.items():
        screen[arm] = {}
        for key in ('admitted_mass', 'retained_benefit', 'missed_benefit', 'retained_harm', 'net_gain', 'benefit_retention'):
            known = [r[key] for r in rows if r[key] is not None]
            screen[arm][key] = dict(mean=float(np.mean(known)) if known else None, defined_groups=len(known), total_groups=108)
    result = dict(groups=108, heads=216, updates=432000, runtime_seconds=done['seconds'],
        first_pair_retrained_exact=True, all108_pairs_retrained=False, occurrence_frozen_exact_all=True,
        input_source='cached_verified', training_source='fresh_run_local_arm64_Torch',
        marginal_fitting_loss={a: float(np.mean(v)) for a, v in loss.items()}, fitting_improved_groups=int(improved),
        fitting_only_sign_screen=screen, independent_arithmetic_checks=checks,
        held_readout='not_run_at_training_freeze', deployment_changed=False, independent_confirmation=False)
    result['fitting_components'] = {a: {k: dict(
        mean=float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None,
        defined_groups=sum(r[k] is not None for r in rows), total_groups=108)
        for k in rows[0]} for a, rows in components.items()}
    run.train.immutable(run.PUBLIC/'training_report.json', result)
    text = f'''# Fixed Occurrence: Paired Training

Fresh native-arm64 Torch training:108 groups,216 heads,432,000 new updates.
The full invocation took {done['seconds']:.2f}s, resuming the pilot group; this
does not include the earlier pilot's elapsed time. All checkpoints
hash-verify. Initial states and sampled queries match between paired arms.
Occurrence parameters and predicted probabilities remain exactly frozen in every
fixed arm. The first full pair was retrained with exact non-timing states; the
other107 pairs were not independently retrained.

Both arms start from the same uncapped checkpoint, split into identical
occurrence/cost branches and reset AdamW. Only occurrence trainability differs.
This is not a test of splitting versus the old shared architecture.

| Arm | Complete fitting marginal objective |
|---|---:|
| Trainable occurrence | {result['marginal_fitting_loss']['trainable']:.9g} |
| Fixed occurrence | {result['marginal_fitting_loss']['fixed']:.9g} |

Fixed occurrence improves the complete fitting objective in{improved}/108 groups.
All occurrence, row/query, conditional reference/harm and signed-bias components
are reported in training_report.json, with undefined values retained. These are
equal-group averages of source/query-balanced quantities; 108 overlapping views
are not 108 independent samples.
The fitting-only sign-screen counts in training_report.json are diagnostics, not
deployment actions: they omit overall risk, eligibility and joint utility.
{checks:,} arithmetic checks verify the source/query-weighted accounting.
Fitting scores are not calibration or transport results; no model or threshold
is selected from them. {BOUNDARIES}
'''
    put('training_report.md', text); print(json.dumps(result)); return result


def readout():
    start = time.monotonic(); run.api.torch.set_num_threads(4); run.api.torch.set_num_interop_threads(1)
    training()
    cfg, ident, data, jobs, oid, pid, fits, actions = run.load()
    details = json.loads((run.PRIVATE/'details.json').read_text()); s = json.loads((run.PUBLIC/'summary.json').read_text())
    check_record_sets(details)
    assert s['identity'] == ident
    for name in ('prediction_replay', 'evaluation_replay'):
        doc = json.loads((run.PUBLIC/(name+'.json')).read_text()); assert doc['exact']
        if name == 'prediction_replay': assert doc['groups'] == 108
        else:
            for k in ('summary', 'details'): assert run.base.artifact(ROOT/doc[k]['path']) == doc[k]
    metric = {(r['group'], r['site'], r['policy']): r['metric'] for r in details['rows']}
    quality = {(r['group'], r['site'], r['policy']): r['metric'] for r in details['qualities']}
    assert len(metric) == len(details['rows']) == 216*len(run.POLICIES)
    assert len(quality) == len(details['qualities']) == 432
    frozen = json.loads((run.PUBLIC/'decision_freeze.json').read_text()); assert frozen['identity'] == ident
    refs = {Path(r['path']).stem: r for r in frozen['groups']}
    assert len(refs) == len(frozen['groups']) == 108
    counts = dict(groups=0, query_constraints=0, changed_queries=0, cost_views=0, quality_views=0, reductions=0)
    for c in run.base.floor_api.contexts(run.causal_view(data), jobs, oid):
        cv, _, (floor, _), (neural, _) = run.base.floor_api.costs(c, data, np.arange(len(c['ids'])))
        for pair in range(6):
            name = c['name']+f'_pair{pair}'; doc = run.checked(refs[name]); fit = run.checked(doc['fit'])
            roles = fit['identity']['source_identity']['roles']
            assert doc['future_fields_removed'] and not doc['held_outcomes_used'] and not fit['held_outcomes_used']
            run.base.floor_api.api.assert_roles(roles['producer_sites'], roles['controller_sites'], roles['training_sites'], roles['held_sites'])
            ref = doc['arrays']; assert run.base.artifact(ROOT/ref['path']) == ref
            with np.load(ROOT/ref['path'], allow_pickle=False) as z: a = {k: z[k].copy() for k in z.files}
            held = np.flatnonzero(np.isin(data['sites'][c['ids']], roles['held_sites']))
            np.testing.assert_array_equal(a['ids'], c['ids'][held])
            state = run.old.api.head.read_checkpoint(ROOT/fit['artifacts']['fixed']['path']); scale = state['warm_start']['norm']['cost_scale']
            ridge = json.loads((run.base.floor_api.PRIVATE/'fits'/name/'complete.json').read_text())
            ref = ridge['artifacts']['scores']; assert run.base.artifact(ROOT/ref['path']) == ref
            with np.load(ROOT/ref['path'], allow_pickle=False) as z:
                np.testing.assert_array_equal(a['ids'], z['ids']); utility = (z['scores'][:, 5].astype(float)-z['scores'][:, 6].astype(float))/scale
            aliases = {k: a[k] for k in ('raw_scores', 'raw_independent', 'raw_joint', 'raw_matched', 'eligible', 'common_anchor')}
            for arm, key in [('trainable', 'uncapped'), ('fixed', 'risk_priority')]:
                for suffix in ('scores', 'independent', 'joint', 'matched'): aliases[key+'_'+suffix] = a[arm+'_'+suffix]
            n, changed = check_actions(aliases, data['sites'][a['ids']], data['recordings'][a['ids']], data['frames'][a['ids']], utility)
            counts['query_constraints'] += n; counts['changed_queries'] += changed
            for site in roles['held_sites']:
                at = data['sites'][a['ids']] == site; ix = held[at]
                for policy in run.POLICIES:
                    take = np.zeros(len(ix), bool) if policy == 'floor' else a[policy][at]
                    m = metric[name, site, policy]; check_selected_costs(m, cv[ix], floor[ix], neural[ix], take)
                    check_fixed_denominator(m, floor[ix], neural[ix], np.isfinite(cv[ix]), take); counts['cost_views'] += 1
                truth = np.column_stack(((cv[ix] > 0) & (cv[ix] <= c['job']['design']['easy_cut']), floor[ix]/scale, np.maximum(neural[ix]-floor[ix], 0)/scale))
                truth[~np.isfinite(cv[ix])] = np.nan
                for arm in run.api.ARMS:
                    check_quality(quality[name, site, arm], a[arm+'_scores'][at], truth, data['recordings'][c['ids'][ix]], data['frames'][c['ids'][ix]])
                    counts['quality_views'] += 1
            counts['groups'] += 1
        run.train.beat('independent_readout_checks', **counts)
    for row in details['contrasts']:
        left, right = row['policy'].split('_vs_')
        check_contrasts(row['metric'], metric[row['group'], row['site'], left], metric[row['group'], row['site'], right])
    for section, key in [('summary', 'rows'), ('paired', 'contrasts'), ('quality', 'qualities')]:
        for policy, metrics in s[section].items():
            rows = [r for r in details[key] if r['policy'] == policy]
            for field, result in metrics.items(): counts['reductions'] += reduce_check(result, rows, field, cfg['bootstrap_seed'], cfg['bootstrap_resamples'])
    for seed, policies in s['by_seed'].items():
        for policy, metrics in policies.items():
            rows = [r for r in details['rows'] if r['seed'] == int(seed) and r['policy'] == policy]
            for field, result in metrics.items(): counts['reductions'] += reduce_check(result, rows, field, cfg['bootstrap_seed'], cfg['bootstrap_resamples'])
    assert s['gates'] == run.gates_for(s['paired'], details['rows'], details['contrasts'])
    proc = subprocess.run([sys.executable, '-m', 'pytest', '-q', *TESTS], cwd=ROOT, capture_output=True, text=True)
    (run.PRIVATE/'verification_pytest.txt').write_text(proc.stdout+proc.stderr)
    if proc.returncode: raise RuntimeError(proc.stdout+proc.stderr)
    count = int(re.search(r'(\d+) passed', proc.stdout).group(1))
    ci = s['paired']['fixed_matched_vs_trainable_matched']['ADE_gain_percent']['ci95']
    verdict = verdict_for(ci, s['gates']['exploratory_screen_pass'])
    lines = ['# Fixed Occurrence: Development Readout', '', f'**Verdict: {verdict}. No deployment change.**', '',
        'Fresh paired local Torch training and readout; cached_verified upstream forecasters/floor/utility. '
        'Actions were committed before outcomes. All108 actions and the numerical readout replay exactly. '
        'Only the first training pair was retrained for exact replay.', '',
        '| Registered contrast | ADE gain % [nominal95% CI] | All-floor harm reduction pp | Selected-risk reduction pp |', '|---|---:|---:|---:|']
    for name, m in s['paired'].items(): lines.append(f"| {name} | {value(m['ADE_gain_percent'])} | {value(m['all_reference_harm_reduction_pp'])} | {value(m['selected_harm_reduction_pp'])} |")
    lines += ['', '| Policy | ADE gain/floor % | Hard gain/floor % | Intervention | Risk violations | Undefined risk | Worst easy gain/CV % |', '|---|---:|---:|---:|---:|---:|---:|']
    for policy in run.POLICIES:
        m, w = s['summary'][policy], s['worst_views'][policy]
        lines.append(f"| {policy} | {value(m['all_gain_floor'])} | {value(m['hard_gain_floor'])} | {value(m['intervention_rate'])} | {w['risk_violating_views']} | {w['undefined_selected_risk_views']} | {w['worst_easy_gain_CV']:.7g} |")
    lines += ['', '## Every Gate', '']+[f'- {k}: {v}.' for k, v in s['gates'].items()]
    lines += ['', '## Boundaries', '', BOUNDARIES]
    put('results.md', '\n'.join(lines)+'\n')
    run.train.immutable(run.PUBLIC/'interpretation.json', dict(verdict=verdict, deployment_changed=False, independent_confirmation=False, submission_ready=False))
    put('verification_report.md', '# Verification Scope\n\n'+json.dumps(counts, indent=2)+f'\n\n{count} tests in{len(TESTS)} files pass. All checkpoints, role exclusions and frozen occurrence parameters checked. Full legacy integration and cold raw rebuild not_run. Engineering verification is not efficacy.\n')
    sources = {**ident['bindings'], **{str(p.relative_to(ROOT)): run.train.digest(p) for p in train_closure()}}
    sources.update({p: run.train.digest(ROOT/p) for p in TESTS})
    artifacts = {p.name: run.train.digest(p) for p in run.PUBLIC.iterdir() if p.suffix in ('.json', '.md') and p.name != 'verification.json'}
    run.train.immutable(run.PUBLIC/'verification.json', dict(source_bindings=sources, artifacts=artifacts,
        details=run.base.artifact(run.PRIVATE/'details.json'), counts=counts, tests=count, test_files=len(TESTS),
        seconds=time.monotonic()-start, deployment_changed=False))
    print(json.dumps(dict(verified=True, verdict=verdict, checks=counts, tests=count)))


def train_closure():
    return run.train.closure(ROOT, ['scripts.verify_m3w_fixed_occurrence'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=['training', 'readout'])
    training() if p.parse_args().phase == 'training' else readout()
