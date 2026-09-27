"""Seal full replay and scoped checks separately from scientific success."""
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_agent_track_refit as run
from scripts.verify_m3w_european_partial_neighbor_refit import TESTS as PARENT_TESTS

TESTS=PARENT_TESTS+['tests/test_m3w_agent_track_refit.py','tests/test_m3w_motion_envelope_diagnostic.py',
                  'tests/test_m3w_motion_unit_sensitivity.py']


def main():
    path=run.PUBLIC/'verification.json'
    if path.exists(): raise ValueError('Validate the existing seal, never replace it')
    cfg,reg,data,jobs,identity=run.load()
    run.endpoints(jobs)
    for name,key,target in [('prediction_replay.json','prediction_freeze_sha256','prediction_freeze.json'),
                            ('evaluation_replay.json','evaluation_sha256','evaluation.json')]:
        d=json.loads((run.PUBLIC/name).read_text())
        assert d['all_exact'] and d[key]==run.digest(run.PUBLIC/target)
    names=['results.md','absolute_costs.md','causal_slices.md','motion_proxies.md',
           'training.md','conclusions.md','matched_topology.png','operations.json','operations.md',
           'association_probe.json','association_probe.md','envelope_diagnostic.json','envelope_diagnostic.md',
           'unit_sensitivity.json','unit_sensitivity.md']
    before={n:run.digest(run.PUBLIC/n) for n in names}
    for script in ('report_m3w_european_agent_track_refit.py','report_m3w_agent_track_operations.py',
                   'probe_m3w_trained_agent_track.py','diagnose_m3w_agent_track_envelope.py',
                   'probe_m3w_motion_unit_sensitivity.py'):
        subprocess.run([sys.executable,'scripts/'+script],cwd=ROOT,check=True)
    assert all(run.digest(run.PUBLIC/n)==h for n,h in before.items())
    home=run.PRIVATE/'verification_runs'/str(time.time_ns()); home.mkdir(parents=True)
    log=home/'tests.log'; xml=home/'tests.xml'
    with log.open('w') as f:
        subprocess.run([sys.executable,'-m','pytest','-q',*TESTS,'--junitxml',str(xml)],cwd=ROOT,
            stdout=f,stderr=subprocess.STDOUT,check=True)
    suite=ET.parse(xml).getroot(); cases=suite.findall('.//testcase')
    assert cases and not any(suite.findall('.//'+k) for k in ('failure','error','skipped'))
    parent=json.loads((run.old.PUBLIC/'verification.json').read_text())
    sources=sorted(set(list(parent['source_bindings'])+run.FILES+TESTS+[
        'scripts/report_m3w_european_agent_track_refit.py','scripts/report_m3w_agent_track_operations.py',
        'scripts/verify_m3w_european_agent_track_refit.py','scripts/probe_m3w_trained_agent_track.py',
        'scripts/diagnose_m3w_agent_track_envelope.py','src/evaluation/m3w_motion_envelope_diagnostic.py',
        'scripts/probe_m3w_motion_unit_sensitivity.py']))
    run.immutable_json(path,dict(new_models=9,updates_per_model=4000,initial_parameters_and_sampling_matched=True,
        nine_pair_full_inference_replay_exact=True,readout_replay_exact=True,reports_figure_byte_reproducible=True,
        scoped_tests_passed=len(cases),scoped_test_files=len(TESTS),
        source_bindings={p:run.digest(ROOT/p) for p in sources},
        artifacts={p.name:run.digest(p) for p in sorted(run.PUBLIC.iterdir()) if p.is_file()},
        logs=[run.artifact(log),run.artifact(xml)],execution_logs=[run.artifact(p) for p in sorted(run.PRIVATE.glob('*.log'))],
        full_legacy_suite='not_run',cold_raw_download_rebuild='not_run',independent_confirmation=False,
        policy_safety_established=False,deployment_changed=False))
    print(json.dumps(dict(tests=len(cases),files=len(TESTS),replay_exact=True,seal=run.artifact(path))))


if __name__=='__main__': main()
