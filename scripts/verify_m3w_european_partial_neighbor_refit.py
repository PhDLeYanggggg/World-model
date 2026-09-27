"""Seal exact full inference/readout replay and scoped tests, not a success claim."""
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_partial_neighbor_refit as run
from scripts.verify_m3w_european_observation_quality import TESTS as INPUT_TESTS

TESTS=INPUT_TESTS+['tests/test_m3w_partial_neighbor_refit.py','tests/test_m3w_native_forecast.py',
                 'tests/test_m3w_native_metrics.py','tests/test_m3w_neighbor_association_probe.py',
                 'tests/test_m3w_agent_track_context.py']


def main():
    dest=run.PUBLIC/'verification.json'
    if dest.exists(): raise ValueError('Validate existing seal instead of replacing it')
    run.load()
    for name,key,target in [('prediction_replay.json','prediction_freeze_sha256','prediction_freeze.json'),
                            ('evaluation_replay.json','evaluation_sha256','evaluation.json')]:
        doc=json.loads((run.PUBLIC/name).read_text())
        assert doc['all_exact'] and doc[key]==run.digest(run.PUBLIC/target)
    frozen=json.loads((run.PUBLIC/'prediction_freeze.json').read_text())
    for ref in frozen['training']:
        assert run.artifact(ROOT/ref['path'])==ref
        cp=json.loads((ROOT/ref['path']).read_text())['checkpoint']
        assert run.artifact(ROOT/cp['path'])==cp
    control=json.loads((run.PUBLIC/'legacy_control.json').read_text())
    assert control['parameters_exact'] and control['sampling_exact']
    assert run.artifact(ROOT/control['complete']['path'])==control['complete']
    names=['results.md','absolute_costs.md','motion_proxies.md','training.md','conclusions.md','gates.json','matched_refit.png',
           'input_slices.json','input_slices.md','association_probe.json']
    before={n:run.digest(run.PUBLIC/n) for n in names}
    subprocess.run([sys.executable,'scripts/report_m3w_european_partial_neighbor_refit.py'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'scripts/diagnose_m3w_european_partial_neighbor_refit.py'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'scripts/probe_m3w_neighbor_association.py'],cwd=ROOT,check=True)
    assert all(run.digest(run.PUBLIC/n)==h for n,h in before.items())
    home=run.PRIVATE/'verification_runs'/str(time.time_ns()); home.mkdir(parents=True)
    log=home/'tests.log'; xml=home/'tests.xml'
    with log.open('w') as out:
        subprocess.run([sys.executable,'-m','pytest','-q',*TESTS,'--junitxml',str(xml)],cwd=ROOT,
            stdout=out,stderr=subprocess.STDOUT,check=True)
    suite=ET.parse(xml).getroot(); cases=suite.findall('.//testcase')
    assert cases and not any(suite.findall('.//'+k) for k in ('error','failure','skipped'))
    observation=json.loads((run.BASE/'european_observation_quality_v1/verification.json').read_text())
    sources=sorted(set(list(observation['source_bindings'])+run.FILES+TESTS+
        ['scripts/report_m3w_european_partial_neighbor_refit.py','scripts/verify_m3w_european_partial_neighbor_refit.py',
         'scripts/diagnose_m3w_european_partial_neighbor_refit.py',
         'scripts/probe_m3w_neighbor_association.py','src/evaluation/m3w_neighbor_association_probe.py',
         'src/world_model/m3w_agent_track_context.py']))
    run.immutable_json(dest,dict(fresh_neural_models=9,updates_per_model=4000,
        legacy_parameter_and_sampler_reproduction_exact=True,
        nine_pair_full_inference_replay_exact=True,readout_replay_exact=True,reports_figure_byte_reproducible=True,
        scoped_tests_passed=len(cases),scoped_test_files=len(TESTS),
        source_bindings={p:run.digest(ROOT/p) for p in sources},
        artifacts={p.name:run.digest(p) for p in sorted(run.PUBLIC.iterdir()) if p.is_file()},
        logs=[run.artifact(log),run.artifact(xml)],
        execution_logs=[run.artifact(p) for p in sorted(run.PRIVATE.glob('*.log'))],
        full_legacy_suite='not_run',independent_confirmation=False,deployment_changed=False))
    print(json.dumps(dict(tests=len(cases),files=len(TESTS),replay_exact=True,seal=run.artifact(dest))))


if __name__=='__main__': main()
