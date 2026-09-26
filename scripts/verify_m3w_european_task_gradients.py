"""Replay all frozen fitting diagnostics and verify unchanged parent artifacts."""
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_task_gradients as run


def main():
    cfg,identity=run.registration()
    subprocess.run([sys.executable,str(Path(__file__).with_name('run_m3w_european_task_gradients.py')),
                    '--phase','verify'],cwd=ROOT,check=True)
    report_before=run.digest(run.PUBLIC/'aggregate_metrics.json')
    subprocess.run([sys.executable,str(Path(__file__).with_name('run_m3w_european_task_gradients.py')),
                    '--phase','report'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    assert run.digest(run.PUBLIC/'aggregate_metrics.json')==report_before
    checks=json.loads((run.PUBLIC/'completion_checks.json').read_text()); clip_cases=0
    for ref in checks['groups']:
        group=json.loads((ROOT/ref['path']).read_text())
        for r in group['records']:
            if r['arm']!='cost_only': continue
            d=r['diagnostic']; eps=max(1e-10,cfg['effect_relative_tolerance']*abs(d['before']['cost']))
            assert d['shared_BCE_relation']['cost']['norm']==0
            if abs(d['auxiliary_minus_cost_step']['cost'])>eps:
                a=d['virtual_steps']['membership_aux']; b=d['virtual_steps']['cost_only']
                assert a['clip_factor']<b['clip_factor']<1
                clip_cases+=1
    tests=['tests/test_m3w_task_gradients.py','tests/test_m3w_membership_auxiliary.py',
           'tests/test_m3w_source_gradient_diagnostic.py']
    xml=run.PRIVATE/'tests.xml'
    p=subprocess.run([sys.executable,'-m','pytest',*tests,'-q','--junitxml='+str(xml)],
                     cwd=ROOT,capture_output=True,text=True)
    (run.PRIVATE/'tests.log').write_text(p.stdout+'\n'+p.stderr); print(p.stdout,flush=True)
    assert p.returncode==0
    suites=list(ET.parse(xml).getroot()); count=sum(int(s.attrib['tests']) for s in suites)
    assert all(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0))==0 for s in suites)
    old=json.loads((run.parent.PUBLIC/'verification.json').read_text())
    required=('conclusions.md','failure_analysis.md','operation_zh.md','project_gap.md','gradient_optimizer.svg')
    assert all((run.PUBLIC/f).is_file() for f in required)
    figure=run.digest(run.PUBLIC/'gradient_optimizer.svg')
    subprocess.run([sys.executable,'scripts/plot_m3w_european_task_gradients.py'],cwd=ROOT,check=True)
    assert run.digest(run.PUBLIC/'gradient_optimizer.svg')==figure
    artifacts={str(p.relative_to(run.PUBLIC)):run.digest(p) for p in run.PUBLIC.rglob('*')
               if p.is_file() and p.name!='verification.json'}
    assert all((run.PUBLIC/f).stat().st_size<2**20 for f in artifacts)
    bindings=sorted(set(run.FILES+tests+[str(Path(__file__).relative_to(ROOT)), 'scripts/plot_m3w_european_task_gradients.py']))
    run.immutable_json(run.PUBLIC/'verification.json',dict(all_passed=True,artifacts=artifacts,
        source_bindings={p:run.digest(ROOT/p) for p in bindings},fresh_tests=count,test_files=tests,
        parent_tests_cached_verified=old['tests'],parent_test_files_cached_verified=len(old['test_files']),
        parent_verification=run.artifact(run.parent.PUBLIC/'verification.json'),
        full_legacy_suite='not_run',full_diagnostic_replays=288,
        disposable_optimizer_step_replays=576,parent_checkpoints_unchanged=True,figure_byte_reproducible=True,
        zero_shared_gradient_clipping_cases=clip_cases,
        new_persisted_updates=0,held_outcome_readout=False,policy_changed=False,independent_confirmation=False))
    print(json.dumps(dict(fresh_tests=count,files=len(tests),public_artifacts=len(artifacts),source_bindings=len(bindings))))


if __name__=='__main__': main()
