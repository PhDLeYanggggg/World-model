"""Post-readout accounting only; future support never changes a decision."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_european_selected_risk_learning as run


def account(moments, targets, bits, groups, sites):
    m, y = np.asarray(moments, float), np.asarray(targets, float)
    known = np.isfinite(y).all(1)
    out = {}
    for site in sorted(set(sites)):
        pop = sites == site; use = pop & known; action = use & bits
        out[str(site)] = dict(rows=int(pop.sum()), supported_rows=int(use.sum()),
            unknown_actions=int((pop & ~known & bits).sum()), events={})
        for name, d, h in (('all', 0, 1), ('easy', 2, 3)):
            mass = float(y[use, d].sum()); harm = float(y[action, h].sum())
            pred_mass = float(m[pop, d].sum()); pred_harm = float(m[pop & bits, h].sum())
            out[str(site)]['events'][name] = dict(actual_reference_mass=mass,
                predicted_reference_mass_all_rows=pred_mass,
                predicted_reference_mass_supported=float(m[use, d].sum()),
                unknown_predicted_mass_fraction=float(m[pop & ~known, d].sum()/pred_mass) if pred_mass > 0 else None,
                actual_selected_harm=harm, predicted_selected_harm_all_rows=pred_harm,
                predicted_selected_harm_supported=float(m[action, h].sum()),
                actual_harm_ratio=harm/mass if mass > 0 else None,
                predicted_harm_ratio=pred_harm/pred_mass if pred_mass > 0 else None,
                active_queries=0, supported_spend_exceeds_supported_predicted_budget=0)
    for q in groups:
        site = str(sites[q[0]]); assert np.all(sites[q] == sites[q[0]])
        if not bits[q].any(): continue
        use = q[known[q]]; action = use[bits[use]]; selected = q[bits[q]]
        for name, d, h in (('all', 0, 1), ('easy', 2, 3)):
            total = .02*m[q, d].sum(); spent = m[selected, h].sum()
            assert spent <= total+1e-10*max(1., total)
            e = out[site]['events'][name]; e['active_queries'] += 1
            e['supported_spend_exceeds_supported_predicted_budget'] += int(
                m[action, h].sum() > .02*m[use, d].sum()+1e-10)
    return out


def main():
    cfg, identity = run.registration(); done = run.checked_training(identity)
    checks = json.loads((run.PUBLIC/'completion_checks.json').read_text()); assert checks['all_passed']
    run.previous.inc.ensure_frozen(); _, _, ctx, _, _, _ = run.previous.inc.load(); data = ctx[2]
    for ref in checks['groups']: assert run.artifact(ROOT/ref['path']) == ref
    values = []; mass_checks = 0
    for ref in done['decisions']:
        receipt = json.loads((ROOT/ref['path']).read_text()); group = receipt['group']; pair = receipt['pair']
        name = group['group']; source = run.previous.PRIVATE/'source'/(name+'_'+pair)
        old = json.loads((source/'receipt.json').read_text())
        assert run.artifact(source/'labels.npz') == old['arrays']['labels']
        with np.load(source/'labels.npz', allow_pickle=False) as z:
            target = run.method.event_targets(z['cv'], z['reference'], z['candidate'], group['easy_cut'])
        with np.load(ROOT/receipt['decisions']['path'], allow_pickle=False) as z:
            ids = z['ids'].copy(); bits = {arm:z[arm+'_joint'].copy() for arm in cfg['arms']}
        assert run.array_hash(ids) == old['ids_sha256']
        sites = data['sites'][ids]
        queries = run.query_groups(sites, data['recordings'][ids], data['frames'][ids])
        for arm in cfg['arms']:
            with np.load(run.PRIVATE/'heads'/(name+'_'+pair+'_'+arm)/'scores.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(ids, z['ids']); moments = z['scores'].copy()
            measured = account(moments, target, bits[arm], queries, sites)
            independent = json.loads((run.PUBLIC/'groups'/(name+'_'+pair+'.json')).read_text())['views'][arm+'_joint']['risk']['by_locality']
            for site, record in measured.items():
                assert record['unknown_actions'] == independent[site]['unknown_selected']
                for event, e in record['events'].items():
                    other = independent[site]['events'][event]
                    np.testing.assert_allclose([e['actual_reference_mass'], e['actual_selected_harm']],
                        [other['reference_mass'], other['positive_harm']], rtol=1e-10, atol=1e-8)
                    mass_checks += 1
            values.append(dict(group=name, pair=pair, arm=arm, accounting=measured))
    run.immutable_json(run.PUBLIC/'query_budget_audit.json', dict(result_source='fresh_run_posthoc_accounting',
        frozen_decisions_changed=False, future_support_in_inference=False,
        counterfactual_not_a_deployable_rule=True, groups=values))
    run.immutable_json(run.PUBLIC/'query_budget_verification.json', dict(all_passed=True,
        independent_locality_event_mass_checks=mass_checks, audit=run.artifact(run.PUBLIC/'query_budget_audit.json'),
        source_sha256=run.digest(Path(__file__)), fresh_run=True, decisions_changed=False))
    print(json.dumps(dict(groups=len(values), mass_checks=mass_checks, decisions_changed=False)))


if __name__ == '__main__': main()
