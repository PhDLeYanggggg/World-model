"""Predeclared mean/unanimity diagnostic before new-seed result exposure."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_donor_seeds as run
from src.evaluation.m3w_seed_consensus import scores

PUBLIC = run.PUBLIC/'consensus'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), ROOT/'src/evaluation/m3w_seed_consensus.py',
         ROOT/'tests/test_m3w_seed_consensus.py', PUBLIC/'protocol.md')


def register():
    if not REG.exists() and (run.PUBLIC/'analysis.json').exists():
        raise ValueError('Do not call this prospectively registered after seeing new-seed results')
    value = dict(files={str(p.relative_to(ROOT)):run.sha(p) for p in FILES},
        parent_registration=run.artifact(run.REG), scores=['mean','unanimity'],
        arm='augmented', seeds=[17,29,43], threshold=0,
        registered_after_seed17_exposure=True, registered_before_seed29_43_readout=True,
        no_threshold_search=True, confidence_coverage_guaranteed=False,
        new_outer_C_H_readout=False, deployment_changed=False)
    run.once(REG, value)


def evaluate(replay):
    register()
    for p in (*FILES, REG):
        content = run.subprocess.run(['git','show','HEAD:'+str(p.relative_to(ROOT))], cwd=ROOT,
                                     capture_output=True, check=True).stdout
        if run.hashlib.sha256(content).hexdigest()!=run.sha(p):
            raise ValueError('Commit fixed ensemble diagnostic before evaluation')
    a = run.read(run.PUBLIC/'analysis.json')
    cfg, parent, previous, frozen = run.register()
    records = []
    for c in previous['contexts']:
        own, _ = run.old.old.load(c)
        for f in c['folds']:
            at = own['sites']==f['source']
            predictions = []
            for seed in (17,29,43):
                name = f'{c["job"]}_{f["source"]}_augmented_seed{seed}.npy'
                p = run.np.load(run.PRIVATE/'predictions'/name, allow_pickle=False)
                row = next(r for r in a['folds'] if r['job']==c['job'] and r['source']==f['source']
                           and r['seed']==seed and r['arm']=='augmented')
                run.method.original.parent.storage.base.exact(run.method.original.parent.digest(p), row['prediction_hash'])
                predictions.append(p)
            s = scores(run.np.stack(predictions))
            record = dict(job=c['job'], source=f['source'], rows=int(at.sum()), arms={})
            for arm, pred in s.items():
                record['arms'][arm] = run.method.original.parent.base.metrics(pred, own['target'][at],
                    own['weights'][at], own['target_scale'][at][0])
            unanimity = (s['unanimity']<=0).all(1)
            if not run.np.array_equal(unanimity, (run.np.stack(predictions)<=0).all(2).all(0)):
                raise ValueError('Unanimity must equal the fixed decision intersection')
            records.append(record)
    contexts = []
    for c in frozen['contexts']:
        rows = [r for r in records if r['job']==c['job']]
        means = {arm:{k:float(run.np.mean([r['arms'][arm][k] for r in rows])) for k in run.METRICS}
                 for arm in ('mean','unanimity')}
        counts = {arm:{k:sum(r['arms'][arm][k] for r in rows) for k in
                        ('accepted_known_proposals','false_safe_count')} for arm in ('mean','unanimity')}
        contexts.append(dict(job=c['job'], means=means, counts=counts, forest=c['source_macro']['shared64'],
            no_full_policy_safety_claim=True))
    result = run.stable(dict(records=records, contexts=contexts, role='fixed_seed_consensus_information_diagnostic',
        no_threshold_search=True, coverage_guaranteed=False, new_outer_C_H_readout=False,
        original_rules_unchanged=True, independent_confirmation=False, identity_certification=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False))
    if replay:
        run.method.original.parent.storage.base.exact(run.read(PUBLIC/'analysis.json'), result)
    else:
        run.once(PUBLIC/'analysis.json', result)
    print('Consensus diagnostic replay exact.' if replay else 'Fixed consensus diagnostic complete; no deployment claim.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['register','evaluate','replay'])
    args = p.parse_args()
    if args.phase=='register':
        register()
        print('Mean/unanimity diagnostic registered before new-seed readout.')
    else:
        evaluate(args.phase=='replay')


if __name__=='__main__':
    main()
