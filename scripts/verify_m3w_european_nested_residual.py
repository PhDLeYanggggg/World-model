"""Full nested-head/probe replay and direct arithmetic checks, without promotion."""
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_european_nested_residual as run
from scripts.diagnose_m3w_european_nested_residual import accounting


def main():
    cfg, identity = run.registration()
    for phase in ('verify_inner', 'verify_probes', 'verify_eval'):
        subprocess.run([sys.executable, 'scripts/run_m3w_european_nested_residual.py', '--phase', phase], cwd=ROOT, check=True)
    before = run.digest(run.PUBLIC/'aggregate_metrics.json')
    subprocess.run([sys.executable, 'scripts/run_m3w_european_nested_residual.py', '--phase', 'report'], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    assert run.digest(run.PUBLIC/'aggregate_metrics.json') == before
    checks = accounting_checks = 0
    diagnostics=json.loads((run.PUBLIC/'residual_error_accounting.json').read_text())
    by_key={(r['tag'],r['variant']):r for r in diagnostics['rows']}
    assert len(by_key)==432
    for g, data, pairs in run.base.contexts(identity['source']):
        bi = pairs['B']['ids']; sites = data['sites'][bi]; cv = data['baseline_ade'][bi, 1]
        for pair in cfg['pairs']:
            _, env, by, *_ = run.base.pair_inputs(g, data, pairs, pair)
            name = g['group']+'_'+pair+'.json'
            result = json.loads((run.PUBLIC/'groups'/name).read_text())
            original = json.loads((run.source.PUBLIC/'groups'/name).read_text())
            for held in sorted(set(sites)):
                te = sites == held; tag = g['group']+'_'+pair+'_'+held
                prior = next(f for f in original['folds'] if f['held'] == held)
                y = run.source.tail.diagnostic.event_targets(by[te], cv[te], prior['easy_cut'])
                record = next(f for f in result['folds'] if f['held'] == held)
                assert run.array_hash(y) == record['target_sha256']
                with np.load(run.parent.frozen_directory(tag,'original')/'scores.npz', allow_pickle=False) as z: frozen = z['scores'].copy()
                with np.load(run.PRIVATE/'probes'/tag/'scores.npz', allow_pickle=False) as z:
                    for variant in cfg['residual_variants']:
                        for probe in cfg['probe_arms']:
                            key = variant+'__'+probe; p = z[key]
                            np.testing.assert_array_equal(p[:,:3],frozen[:,:3])
                            assert np.isfinite(p).all() and (p[:,3]>=0).all() and (p[:,3]<=p[:,1]).all()
                            for subset, use in (('all',np.ones(len(y),bool)), ('envelope_positive',env[te]>0)):
                                use &= np.isfinite(y).all(1)
                                actual = np.mean((p[use,3]-y[use,3])**2)
                                expected = record['metrics'][key][subset]['harm_MSE']
                                np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12); checks += 1
                                if subset=='envelope_positive' and probe=='context_bias':
                                    terms=accounting(frozen[:,3],p[:,3],y[:,3],env[te]>0)
                                    assert all(by_key[(tag,variant)][k]==v for k,v in terms.items())
                                    accounting_checks+=1
    assert checks == 1728
    assert accounting_checks == 432
    tests = ['tests/test_m3w_nested_residual.py', 'tests/test_m3w_context_residual.py',
             'tests/test_m3w_membership_auxiliary.py', 'tests/test_m3w_severity_auxiliary.py',
             'tests/test_m3w_nested_residual_reporting.py', 'tests/test_m3w_nested_residual_accounting.py']
    xml = run.PRIVATE/'tests.xml'
    p = subprocess.run([sys.executable,'-m','pytest',*tests,'-q','--junitxml='+str(xml)],cwd=ROOT,capture_output=True,text=True)
    (run.PRIVATE/'tests.log').write_text(p.stdout+'\n'+p.stderr); print(p.stdout,flush=True); assert p.returncode == 0
    suites = list(ET.parse(xml).getroot()); count = sum(int(s.attrib['tests']) for s in suites)
    assert all(int(s.attrib.get('errors',0))+int(s.attrib.get('failures',0)) == 0 for s in suites)
    for f in ('conclusions.md','failure_analysis.md','project_gap.md','operation_zh.md','nested_residual.svg'):
        assert (run.PUBLIC/f).is_file()
    before = run.digest(run.PUBLIC/'nested_residual.svg')
    subprocess.run([sys.executable,'scripts/plot_m3w_european_nested_residual.py'],cwd=ROOT,check=True)
    assert run.digest(run.PUBLIC/'nested_residual.svg') == before
    files = {str(p.relative_to(run.PUBLIC)):run.digest(p) for p in run.PUBLIC.rglob('*') if p.is_file() and p.name != 'verification.json'}
    assert all((run.PUBLIC/f).stat().st_size < 2**20 for f in files)
    bindings = sorted(set(run.FILES+tests+['scripts/plot_m3w_european_nested_residual.py',
        'scripts/diagnose_m3w_european_nested_residual.py',str(Path(__file__).relative_to(ROOT))]))
    run.immutable_json(run.PUBLIC/'verification.json',dict(all_passed=True,artifacts=files,
        source_bindings={f:run.digest(ROOT/f) for f in bindings},inner_checkpoint_prediction_replays=432,
        closed_form_fit_prediction_replays=864,readout_groups_replayed=36,direct_MSE_checks=checks,
        descriptive_error_decompositions_replayed=accounting_checks,
        fresh_tests=count,test_files=tests,parent25_tests_cached_verified=True,ancestor509_tests_cached_verified=True,
        full_legacy_suite='not_run',figure_byte_reproducible=True,parent_checkpoints_unchanged=True,
        independent_calibration=False,independent_confirmation=False,deployment_changed=False,
        new_neural_risk_updates=864000,new_trajectory_updates=0))
    print(json.dumps(dict(fresh_tests=count,files=len(tests),artifacts=len(files),bindings=len(bindings),
        inner_replays=432,closed_form_replays=864,direct_MSE_checks=checks)))


if __name__ == '__main__': main()
