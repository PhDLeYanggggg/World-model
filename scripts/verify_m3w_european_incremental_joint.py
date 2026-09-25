"""Independent action/cap replay, scoped regression tests and public artifact manifest."""
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_european_incremental_joint as run
from scripts.report_m3w_european_floor_relative import dump


def reference_costs(utility, risk, scale):
    # Production converts stored float32 predictions before arithmetic. Replaying
    # in float32 first changes small budgets; preserve the same source precision.
    u, r = np.asarray(utility, dtype=np.float64), np.asarray(risk, dtype=np.float64)
    return (u[:, 0]-u[:, 1])/scale, r[:, 1]/scale


def replay_actions():
    cfg, _, ctx, _, _, identity, _ = run.load(); data = ctx[2]
    frozen = run.ensure_frozen(); assert frozen['identity'] == identity
    counts = dict(groups=0, policies=0, queries=0, cap_checks=0, full_add_exact=0, original_actions_preserved=0)
    for ref in frozen['groups']:
        r = json.loads((ROOT/ref['path']).read_text())
        with np.load(ROOT/r['artifacts']['choices']['path'], allow_pickle=False) as z: a = {k: z[k].copy() for k in z.files}
        ids = a['ids']; qq = json.loads((ROOT/r['artifacts']['queries']['path']).read_text())
        with np.load(run.parent.PRIVATE/'decisions'/(r['group']+'.npz'), allow_pickle=False) as z:
            pos = np.searchsorted(z['ids'], ids); np.testing.assert_array_equal(z['ids'][pos], ids)
            old = z['old_stop'][pos]; full = z['add_only'][pos]
            u, risk = z['incumbent_reference__utility'][pos], z['incumbent_reference__risk'][pos]
        np.testing.assert_array_equal(a['old_stop'], old); np.testing.assert_array_equal(a['full_add'], full)
        assert len(qq) == 384; counts['full_add_exact'] += 1
        pool = full & ~old; scale = r['cost_scale']
        gain, harm = reference_costs(u, risk, scale)
        for p in cfg['policies']:
            assert not np.any(old & ~a[p]) and not np.any(a[p] & ~full)
            counts['policies'] += 1; counts['original_actions_preserved'] += 1
        for q in qq:
            loc = np.flatnonzero((data['recordings'][ids] == q['recording']) & (data['frames'][ids] == q['frame']))
            assert len(loc) == q['agents']
            eligible = [j for j in loc if pool[j]]; k = len(eligible)//2
            ranked = sorted(eligible, key=lambda j: (-gain[j], ids[j]))[:k]
            refbits = np.zeros(len(loc), bool)
            refbits[np.isin(loc, ranked)] = True
            np.testing.assert_array_equal(a['half_independent'][loc] & ~old[loc], refbits)
            cap = float(harm[ranked].sum()/len(loc))
            np.testing.assert_allclose(cap, q['predicted_harm_budget'], rtol=1e-12, atol=1e-12)
            for p in ('half_independent', 'half_hash', 'half_unary', 'half_joint'):
                chosen = a[p][loc] & ~old[loc]
                assert int(chosen.sum()) == k and not np.any(chosen & ~pool[loc])
                assert float(harm[loc][chosen].sum()/len(loc)) <= cap+1e-10
                counts['cap_checks'] += 1
            if not q['matched']:
                for p in ('half_hash', 'half_unary', 'half_joint'):
                    np.testing.assert_array_equal(a[p][loc], a['half_independent'][loc])
            counts['queries'] += 1
        counts['groups'] += 1
    assert counts == dict(groups=36, policies=252, queries=13824, cap_checks=55296,
                         full_add_exact=36, original_actions_preserved=252)
    return counts


def main():
    counts = replay_actions()
    old = json.loads((run.parent.PUBLIC/'completion_checks.json').read_text())
    files = sorted(set(old['test_files']) | {'tests/test_m3w_incremental_joint.py',
        'tests/test_m3w_incremental_joint_reporting.py', 'tests/test_m3w_native_joint_controls.py',
        'tests/test_m3w_joint_intervention.py', 'tests/test_m3w_incremental_joint_accounting.py'})
    xml = run.PRIVATE/'tests.xml'
    result = subprocess.run([sys.executable, '-m', 'pytest', *files, '-q', '--junitxml='+str(xml)],
        cwd=ROOT, capture_output=True, text=True)
    (run.PRIVATE/'tests.log').write_text(result.stdout+'\n'+result.stderr)
    print(result.stdout[-3000:], flush=True)
    if result.returncode: raise RuntimeError('Scoped tests failed; no completion manifest')
    root = ET.parse(xml).getroot(); suites = list(root) if root.tag == 'testsuites' else [root]
    count = sum(int(s.attrib['tests']) for s in suites)
    assert all(int(s.attrib.get('failures', 0)) == int(s.attrib.get('errors', 0)) == 0 for s in suites)
    compute = json.loads((run.PUBLIC/'compute_receipt.json').read_text())
    assert compute['counts'] == dict(coordinate_arrays=144, metric_reductions=4176)
    required = ('conclusions.md', 'failure_analysis.md', 'project_gap.md', 'model_card.md', 'data_card.md', 'operation_zh.md')
    assert all((run.PUBLIC/f).is_file() for f in required)
    artifacts = {str(f.relative_to(run.PUBLIC)): run.digest(f) for f in run.PUBLIC.rglob('*')
        if f.is_file() and f.name != 'completion_checks.json'}
    assert all((run.PUBLIC/f).stat().st_size < 1024**2 for f in artifacts)
    bindings = {f: run.digest(ROOT/f) for f in (*run.FILES,
        'scripts/report_m3w_european_incremental_joint.py', 'scripts/verify_m3w_european_incremental_joint.py',
        'scripts/diagnose_m3w_european_incremental_joint.py', *files)}
    dump(run.PUBLIC/'completion_checks.json', dict(all_passed=True, action_checks=counts,
        metric_checks=compute['counts'], artifact_hashes=artifacts, source_bindings=bindings,
        test_files=files, test_count=count, scoped_test_files=len(files), full_legacy_suite='not_run',
        decisions=run.artifact(run.PRIVATE/'decisions_complete.json'), evaluation=run.artifact(run.PRIVATE/'evaluation_complete.json')))
    print(json.dumps(dict(tests=count, scoped_files=len(files), artifacts=len(artifacts), counts=counts)))


if __name__ == '__main__': main()
