"""Lossless compact archive and descriptive frozen-TRAIN diagnosis export."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_selected_cost_audit_v1'


def sha(content):
    return hashlib.sha256(content).hexdigest()


def median(records, pool, key):
    values = [r['diagnosis'][pool][key] for r in records if r['diagnosis'][pool][key] is not None]
    return statistics.median(values) if values else None


def summary(records):
    output = {}
    for family in ('no_scene', 'scene'):
        output[family] = {}
        for arm in ('physical_only', 'physical_neural', 'neural_only'):
            rows = [r for r in records if r['family']==family and r['arm']==arm]
            if len(rows) != 444:
                raise ValueError('All444fitting record/head views required per bank/family')
            pools = {}
            for pool in ('all_legal', 'gate_passing', 'selected'):
                pools[pool] = dict(
                    action_views=sum(r['diagnosis'][pool]['action_views'] for r in rows),
                    known_action_views=sum(r['diagnosis'][pool]['known_action_views'] for r in rows),
                    unknown_action_views=sum(r['diagnosis'][pool]['unknown_action_views'] for r in rows),
                    median={key:median(rows,pool,key) for key in ('predicted_all_risk_on_known',
                        'observed_all_risk_on_known', 'predicted_easy_risk_on_known', 'observed_easy_risk_on_known',
                        'harm_prediction_ratio', 'easy_harm_prediction_ratio', 'reference_inflation_ratio',
                        'easy_reference_inflation_ratio')})
            violations = {}
            for event, score_key in (('all','known_positive_risk'), ('easy','known_easy_positive_risk')):
                failed = [r for r in rows if r['original_score'][score_key] is not None
                          and r['original_score'][score_key] > .02+1e-12]
                parts = [r['diagnosis']['selected'][event+'_excess_decomposition'] for r in failed]
                violations[event] = dict(known_risk_violations=len(failed),
                    harm_component_dominates=sum(p['harm_underprediction'] >= p['reference_inflation'] for p in parts),
                    harm_component_mass=sum(p['harm_underprediction'] for p in parts),
                    reference_component_mass=sum(p['reference_inflation'] for p in parts),
                    median_harm_prediction_ratio=median(failed,'selected','harm_prediction_ratio' if event=='all' else 'easy_harm_prediction_ratio'),
                    median_reference_inflation_ratio=median(failed,'selected','reference_inflation_ratio' if event=='all' else 'easy_reference_inflation_ratio'))
            output[family][arm] = dict(record_views=len(rows), pools=pools, violations=violations)
    return output


def render(value):
    lines = ['# Selected Costs: Easy Harm, Not Uniform Harm Bias', '',
        'Descriptive TRAIN diagnosis of108frozen states/2,664record views,zero updates.',
        'Every original action hash and per-record score is replayed. This export',
        'summarizes verified frozen inference;it does not train or calibrate a policy.', '',
        '## Mixed-Bank Pools', '',
        'Medians use finite record-level ratios,not pooled windows. Repeated record/',
        'context/head views are not independent recordings. All-legal and gate pools',
        'repeat candidate rows and are not deployment policies.', '',
        '| Family | Pool | Action views (unknown) | Median known all risk | Median known easy risk | Median predicted/true H | Median predicted/true EH |',
        '|---|---|---:|---:|---:|---:|---:|']
    for family, banks in value['families'].items():
        for pool, row in banks['physical_neural']['pools'].items():
            m = row['median']; fmt = lambda x:'undefined' if x is None else f'{x:.6f}'
            lines.append(f"| {family} | {pool} | {row['action_views']} ({row['unknown_action_views']}) | "
                +' | '.join(fmt(m[k]) for k in ('observed_all_risk_on_known','observed_easy_risk_on_known',
                    'harm_prediction_ratio','easy_harm_prediction_ratio'))+' |')
    lines += ['', '## Accounting On Known Violations', '',
        'Observed minus predicted risk excess = harm underprediction + reference',
        'inflation,with analogous EH/ER terms. Negative components alleviate error.', '',
        '| Family | Event | Violating views | Harm component dominates | Median predicted/true harm | Median predicted/true reference |',
        '|---|---|---:|---:|---:|---:|']
    for family, banks in value['families'].items():
        for event, row in banks['physical_neural']['violations'].items():
            lines.append(f"| {family} | {event} | {row['known_risk_violations']} | {row['harm_component_dominates']} | "
                f"{row['median_harm_prediction_ratio']:.6f} | {row['median_reference_inflation_ratio']:.6f} |")
    lines += ['', '## Interpretation', '',
        'Global harm underprediction is not the universal failure:the selected',
        'mixed-bank median H prediction is above its observed mass. Easy harm has',
        'a different pattern. Do not repair every cost by a generic multiplier.',
        'The exact decomposition measures where prediction error enters the gate;',
        'it does not prove selection caused the error or identify its training cause.',
        'Composition,rare easy events,small support and model misspecification remain.', '',
        'The next single-factor training test targets the sparse easy-harm loss',
        'gradient,retaining architecture,inputs,draws,optimizer and2%evaluation rule.',
        'It is a hypothesis,not a passed gate. The known-label ratios here do not',
        'replace full-completion bounds. Empty risk stays not_estimable. Independent',
        'roles and deployment remain unchanged;no Stage5C orSMC.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--check', action='store_true')
    parser.add_argument('--restore', action='store_true'); args = parser.parse_args()
    raw = PUBLIC/'diagnosis.json'; archive = PUBLIC/'diagnosis.json.gz'; manifest = PUBLIC/'diagnosis_archive.json'
    content = raw.read_bytes() if raw.exists() else gzip.decompress(archive.read_bytes())
    if manifest.exists():
        doc = json.loads(manifest.read_text())
        if sha(content) != doc['uncompressed_sha256'] or sha(archive.read_bytes()) != doc['archive_sha256']:
            raise ValueError('Exact diagnosis archive identity differs')
    elif args.check:
        raise ValueError('Build lossless archive before checking')
    else:
        zipped = gzip.compress(content, mtime=0); archive.write_bytes(zipped)
        manifest.write_text(json.dumps(dict(uncompressed_sha256=sha(content), archive_sha256=sha(zipped),
            uncompressed_bytes=len(content), archive_bytes=len(zipped), numerical_JSON_bytes_unchanged=True,
            raw_data_or_weights_included=False), indent=2)+'\n')
    if args.restore and not raw.exists():
        raw.write_bytes(content)
    verification = json.loads((PUBLIC/'verification.json').read_text())
    if not verification['exact_replay'] or verification['diagnosis_sha256'] != sha(content):
        raise ValueError('Verified frozen inference diagnosis required')
    data = json.loads(content)
    value = dict(result_source='cached_verified_frozen_TRAIN_diagnosis_descriptive_export',
        diagnosis_sha256=sha(content), families=summary(data['records']), optimizer_updates=0,
        independence=False, causal_attribution=False, risk_budget=.02, independent_roles_opened=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    for path, text in {PUBLIC/'summary.json':json.dumps(value, indent=2)+'\n', PUBLIC/'readout.md':render(value)}.items():
        if args.check or path.exists():
            if path.read_text() != text:
                raise ValueError('Preserve derived diagnosis revision')
        else:
            path.write_text(text)
    print(json.dumps(dict(exact_check=args.check, archive_bytes=archive.stat().st_size, fresh_training=False)))


if __name__ == '__main__':
    main()
