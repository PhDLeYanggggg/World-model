"""Verify and summarize fixed TRAIN aggregation evidence without new inference."""
import argparse
import json

from scripts import diagnose_m3w_risk_aggregation as audit


def validate():
    home, sha = audit.PUBLIC, audit.base.sha
    reg = json.loads(audit.REG.read_text())
    complete = json.loads((home/'complete.json').read_text())
    summary = json.loads((home/'summary.json').read_text())
    if (complete['summary_sha256'] != sha(home/'summary.json')
            or complete['registration_sha256'] != sha(audit.REG)
            or reg['parent_freeze_sha256'] != sha(audit.training.PUBLIC/'training_freeze.json')
            or complete['optimizer_steps'] != 0 or complete['validation_read']
            or complete['independent_roles_read'] or complete['deployment_changed']):
        raise ValueError('Incomplete or changed TRAIN-only evidence')
    for rel, expected in reg['bindings'].items():
        if sha(audit.ROOT/rel) != expected: raise ValueError('Changed diagnostic source')
    train = json.loads(audit.training.REG.read_text())
    expected = {(r['group'], r['source'], r['seed'], mode)
                for r in train['expected'] for mode in ('unit_control','budget_scaled')}
    freeze = json.loads((audit.training.PUBLIC/'training_freeze.json').read_text())
    payloads = {}
    for ref in freeze['fits']:
        path = audit.ROOT/ref['path']
        if sha(path) != ref['sha256']: raise ValueError('Changed fit receipt')
        doc = json.loads(path.read_text()); i = doc['identity']
        payloads[i['group'], i['source'], i['seed'], doc['mode']] = doc['checkpoint']['sha256']
    seen = set(); rows = []
    for ref in complete['groups']:
        path = audit.ROOT/ref['path']
        if path.is_symlink() or not path.resolve().is_relative_to((home/'groups').resolve()) or sha(path) != ref['sha256']:
            raise ValueError('Changed group')
        row = json.loads(path.read_text()); i = row['identity']; key = i['group'],i['source'],i['seed'],row['mode']
        if (key not in expected or key in seen or row['checkpoint_sha256'] != payloads[key]
                or row['registration_sha256'] != sha(audit.REG) or not row['monitor_replayed']
                or row['validation_read'] or row['independent_roles_read'] or row['new_training']):
            raise ValueError('Invalid diagnostic identity or role')
        seen.add(key); rows.append(row)
    if seen != expected or len(seen) != 144: raise ValueError('All144 required')
    for mode in ('unit_control','budget_scaled'):
        selected = [r for r in rows if r['mode'] == mode]
        if summary[mode]['views'] != 72: raise ValueError('Wrong view count')
        for gate in ('soft','hard'):
            for risk in ('all','easy'):
                records = [r['aggregations'][gate]['constraints'][risk] for r in selected]
                counts = dict(query_excess_views=sum(r['query_mean_excess'] > 0 for r in records),
                    mass_excess_views=sum(r['agent_mass_mean_excess'] > 0 for r in records),
                    query_pass_mass_fail_views=sum(r['query_mean_excess'] <= 0 < r['agent_mass_mean_excess'] for r in records),
                    mass_pass_query_fail_views=sum(r['agent_mass_mean_excess'] <= 0 < r['query_mean_excess'] for r in records),
                    unsupported_reference_views=sum(r['agent_mass_risk'] is None for r in records))
                if counts != summary[mode][gate][risk]: raise ValueError('Summary count mismatch')
    return complete, summary


def content(complete, summary):
    table = ['| Arm | Decision | Risk | Query excess /72 | Agent-mass excess /72 | Query pass, mass fail /72 | Reverse /72 | Undefined reference /72 |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for arm in ('unit_control','budget_scaled'):
        for gate in ('soft','hard'):
            for risk in ('all','easy'):
                r = summary[arm][gate][risk]
                table.append('| '+arm+' | '+gate+' | '+risk+' | '+' | '.join(str(r[k]) for k in
                    ('query_excess_views','mass_excess_views','query_pass_mass_fail_views',
                     'mass_pass_query_fail_views','unsupported_reference_views'))+' |')
    return f'''# TRAIN Aggregation Diagnostic: Complete

Fresh_run: 144 frozen-model TRAIN-monitor recomputations in {complete['seconds']:.2f}s.
Model checkpoints and source packets are cached_verified. All input hashes,
causal eligibility, fixed monitor draws and saved soft-excess values replay.
No optimizer step, development-label access or independent-role access occurred.

{chr(10).join(table)}

Each view uses the same fixed 128 known-TRAIN query draws as training monitoring.
These are repeated query occurrences, not independent observations or a full
TRAIN census. The agent-mass calculation pools selected harm and reference on
these draws; it is not a new development evaluation or missing-label completion.
The original 2% budget is unchanged. Zero reference is undefined, not safe.

Query pass/mass fail directly exhibits different feasibility judgments under the
same predictions and action weights. The reverse can also occur. It does not
show how much development failure the mismatch causes. Soft/hard mismatch,
unknown outcomes, model error and domain shift remain separate limitations.
No arm is selected or promoted by this diagnostic. Positive scaling preserves
the query constraint zero set, not equivalence with pooled risk. Any new training
repair must be separately registered and retain the scientific readout definition.

Recheck aggregates without inference or training:

```bash
.venv-pytorch/bin/python -m scripts.report_m3w_risk_aggregation --check
```

The project remains dataset-local, raw-frame and 2.5D; no true3D, foundation,
metric/seconds or independent generalization claim follows. Stage5C and SMC stay
disabled. A risk-proxy diagnosis is not new trajectory dynamics.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--check',action='store_true')
    args = parser.parse_args(); text = content(*validate()); path = audit.PUBLIC/'conclusions.md'
    if args.check or path.exists():
        if path.read_text() != text: raise ValueError('Preserve conflicting report')
    else:
        with path.open('x') as file: file.write(text)
    print(json.dumps(dict(status='checked' if args.check else 'reported', new_predictions=False)))


if __name__ == '__main__': main()
