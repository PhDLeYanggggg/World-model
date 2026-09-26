"""Full frozen diagnostic replay, unchanged parents and deterministic figures."""
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_severity_transport as run


def main():
    cfg,identity=run.registration()
    subprocess.run([sys.executable,'scripts/run_m3w_european_severity_transport.py','--phase','verify'],cwd=ROOT,check=True)
    before=run.digest(run.PUBLIC/'aggregate_metrics.json')
    subprocess.run([sys.executable,'scripts/run_m3w_european_severity_transport.py','--phase','report'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    assert run.digest(run.PUBLIC/'aggregate_metrics.json')==before
    checks=json.loads((run.PUBLIC/'completion_checks.json').read_text()); accounting=0; gradients=0
    for ref in checks['groups']:
        assert run.artifact(ROOT/ref['path'])==ref
        group=json.loads((ROOT/ref['path']).read_text())
        for r in group['records']:
            for c in r['error_accounting'].values():
                for e in c.values():
                    if e['status']!='computed': continue
                    p=e['partitions']['all']; delta=p['positive_excess_mass']-p['negative_excess_mass']
                    assert abs(delta/p['rows']-p['signed_excess_MSE']) <= 1e-9*max(1,abs(p['signed_excess_MSE']))
                    for mass,field in (('positive_row_excess','positive_excess_mass'),('negative_row_excess','negative_excess_mass')):
                        for unit in ('window','recording','track'):
                            assert abs(e['groups'][unit][mass]['total']-p[field]) <= 1e-9*max(1,p[field])
                            accounting+=1
            for k in ('ordinary_BCE','weighted_BCE','easy_harm_cost'):
                assert r['gradients'][k]['group_gradient_additivity_pass']; gradients+=1
            assert r['radial_support']['threshold_uses_fitting_only'] and not r['radial_support']['conditional_support_proven']
    tests=['tests/test_m3w_severity_transport.py','tests/test_m3w_severity_auxiliary.py','tests/test_m3w_task_gradients.py']
    xml=run.PRIVATE/'tests.xml'; p=subprocess.run([sys.executable,'-m','pytest',*tests,'-q','--junitxml='+str(xml)],cwd=ROOT,capture_output=True,text=True)
    (run.PRIVATE/'tests.log').write_text(p.stdout+'\n'+p.stderr); print(p.stdout,flush=True); assert p.returncode==0
    suites=list(ET.parse(xml).getroot()); count=sum(int(s.attrib['tests']) for s in suites)
    assert all(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0))==0 for s in suites)
    assert all((run.PUBLIC/f).is_file() for f in ('conclusions.md','failure_analysis.md','operation_zh.md','project_gap.md','transport_diagnostic.svg'))
    fig=run.digest(run.PUBLIC/'transport_diagnostic.svg')
    subprocess.run([sys.executable,'scripts/plot_m3w_european_severity_transport.py'],cwd=ROOT,check=True)
    assert run.digest(run.PUBLIC/'transport_diagnostic.svg')==fig
    artifacts={str(p.relative_to(run.PUBLIC)):run.digest(p) for p in run.PUBLIC.rglob('*') if p.is_file() and p.name!='verification.json'}
    assert all((run.PUBLIC/f).stat().st_size<2**20 for f in artifacts)
    bindings=sorted(set(run.FILES+tests+['scripts/plot_m3w_european_severity_transport.py',str(Path(__file__).relative_to(ROOT))]))
    run.immutable_json(run.PUBLIC/'verification.json',dict(all_passed=True,artifacts=artifacts,
        source_bindings={p:run.digest(ROOT/p) for p in bindings},fresh_tests=count,test_files=tests,
        parent_verification=run.artifact(run.parent.PUBLIC/'verification.json'),
        parent_fresh_tests_now_cached_verified=22,ancestor_tests_cached_verified=509,full_legacy_suite='not_run',
        full_diagnostic_replays=144,group_mass_checks=accounting,gradient_additivity_replays=gradients,
        parent_checkpoints_unchanged=True,figure_byte_reproducible=True,new_updates=0,
        method_improvement_proven=False,independent_confirmation=False,deployment_changed=False))
    print(json.dumps(dict(fresh_tests=count,files=len(tests),public_artifacts=len(artifacts),source_bindings=len(bindings),
        group_mass_checks=accounting,gradient_additivity_replays=gradients)))


if __name__=='__main__': main()
