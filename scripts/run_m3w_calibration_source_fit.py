"""Frozen SOURCE fit diagnosis after unsupported calibration; no new role access."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch imports')
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_selection_cardinal_cost as parent
from scripts import run_m3w_selection_cost_transport as transport
from src.evaluation import m3w_calibration_source_fit as api

prior = parent.prior
NAME = 'european_calibration_source_fit_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_calibration_source_fit_v1.json'; REG = PUBLIC/'registration_v2.json'
sha, checked, artifact, once, write = parent.sha, parent.checked, parent.artifact, parent.once, parent.write
FILES = (Path(__file__), CONFIG, ROOT/'src/evaluation/m3w_calibration_source_fit.py',
         ROOT/'tests/test_m3w_calibration_source_fit.py', PUBLIC/'protocol.md')


def admission(register=False):
    cfg = json.loads(CONFIG.read_text()); _, producers, ids = parent.admission()
    if (cfg['families'] != list(api.FAMILIES) or cfg['expected_model_states'] != 18
            or cfg['expected_site_groups'] != 144 or len(ids) != cfg['source_whitelist_rows']
            or cfg['positive_harm_budget'] != .02 or cfg['seeds'] != [17, 29, 43]
            or any(cfg[k] for k in ('new_training', 'new_normalization', 'new_policy',
                'calibration_array_access', 'confirmation_access', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Complete SOURCE-only frozen diagnosis required')
    old, new = prior.fitted(), parent.fitted()
    models = {(r['trial'], f):(old[r['trial'], 'context'] if f == 'weighted_centered'
              else new[r['trial'], 'mean_absolute']) for r in producers for f in api.FAMILIES}
    calibration = ROOT/'outputs/publication_readiness_2026_09/european_independent_risk_calibration_v1'
    replay = json.loads((calibration/'verify.json').read_text())
    if not replay['all47736_metric_groups_exact'] or replay['summary_sha256'] != sha(calibration/'summary.json'):
        raise ValueError('Completed unchanged calibration result required')
    definition = dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        pre_execution_definition_draft=artifact(PUBLIC/'registration.json'),
        pre_execution_clarification='SOURCE_labels_already_seen_not_inference_inputs_explicit_case_cache_accounting',
        source_registration=artifact(prior.REG), source_training=artifact(prior.PUBLIC/'training_freeze.json'),
        cardinal_training=artifact(parent.PUBLIC/'training_freeze.json'),
        source_bank_freeze=artifact(prior.PUBLIC/'source_bank_freeze.json'),
        old_source_verification=artifact(transport.PUBLIC/'verification.json'),
        motivation_after_calibration=artifact(calibration/'summary.json'),
        calibration_cost_diagnosis=artifact(calibration/'selected_cost_attribution.json'),
        source_only_normalization=True, source_resubstitution_not_validation=True,
        checkpoints={t+'_'+f:models[t, f]['checkpoint'] for t, f in models},
        dependencies={str(p.relative_to(ROOT)):sha(p) for p in
            (Path(transport.__file__), ROOT/'src/evaluation/m3w_selection_cost_transport.py',
             ROOT/'src/evaluation/m3w_selected_cost_diagnosis.py')},
        calibration_arrays_read=False, new_optimizer_updates=0, confirmation_access=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    if register:
        once(REG, definition)
    if json.loads(REG.read_text()) != definition:
        raise ValueError('Frozen diagnostic binding changed')
    if not register:
        for p in (*FILES, REG):
            original = subprocess.check_output(['git', 'show', 'HEAD:'+str(p.relative_to(ROOT))], cwd=ROOT)
            if hashlib.sha256(original).hexdigest() != sha(p):
                raise ValueError('Commit diagnostic before real inference')
    return cfg, producers, ids, models


def execute(phase, resume):
    cfg, producers, whitelist, models = admission()
    pilot, verify = phase == 'pilot', phase == 'verify'
    finish = PUBLIC/('pilot.json' if pilot else 'verification.json' if verify else 'complete.json')
    if finish.exists():
        raise ValueError('Completed phase immutable')
    if not pilot and not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Actual SOURCE resource pilot required')
    torch.set_num_threads(4); torch.set_num_interop_threads(1); started = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume from completed case')))
    def guard():
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
            raise OSError('Preserve 10GiB reserve')
        if time.monotonic()-started > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h boundary; resume unchanged cases')
    def beat(**v):
        guard(); row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), phase=phase, **v)
        write(PRIVATE/'heartbeat.json', row)
        print(json.dumps(row), flush=True)
    beat(state='runtime_ready', architecture=platform.machine(), cpu_threads=4, workers=0, new_training=False)
    source = prior.source_inputs(json.loads(prior.frozen.REG.read_text())); cases = []
    fresh_cases, cached_cases = 0, 0
    for producer in producers[:1] if pilot else producers:
        data = None
        for family in api.FAMILIES:
            path = PUBLIC/'cases'/(producer['trial']+'_'+family+'.json')
            if path.exists() and resume and not verify:
                case = json.loads(path.read_text())
                if (case['registration_sha256'] != sha(REG) or case['trial'] != producer['trial']
                        or case['family'] != family or case['checkpoint'] != models[producer['trial'], family]['checkpoint']):
                    raise ValueError('Resume binding changed')
                if (PUBLIC/'pilot.json').exists():
                    refs = json.loads((PUBLIC/'pilot.json').read_text())['cases']
                    for ref in refs:
                        if ROOT/ref['path'] == path:
                            checked(ref)
                cached_cases += 1
                cases.append(case); beat(state='case_cached_verified', completed=len(cases)); continue
            guard(); beat(state='SOURCE_case', trial=producer['trial'], family=family)
            if data is None:
                data = transport.source_case(producer, source, whitelist)
            causal = data['causal']; ref = models[producer['trial'], family]
            state = prior.api.read_checkpoint(checked(ref['checkpoint']))
            np.testing.assert_array_equal(state['fit_ids'], causal['ids'])
            prediction = prior.api.predict(state, causal)
            choices, _ = prior.allocator.select_actions(prediction, causal['allowed'], arm='physical_neural')
            groups = []
            for site in sorted(set(causal['sites'])):
                at = np.flatnonzero(causal['sites'] == site)
                diagnosis = transport.api.diagnose(prediction[at], data['target'][at], causal['allowed'][at],
                    choices[at], causal['scales'][at])
                records = {str(rec):transport.api.selected_summary(prediction[ix], data['target'][ix],
                    causal['allowed'][ix], choices[ix], causal['scales'][ix])
                    for rec in sorted(set(data['recordings'][at])) for ix in [at[data['recordings'][at] == rec]]}
                hashes = dict(prediction_sha256=prior.frozen.fingerprint(prediction[at]),
                    target_cost_sha256=prior.frozen.fingerprint(data['target'][at]),
                    choices_sha256=prior.frozen.fingerprint(choices[at]))
                old_exact = None
                if family == 'weighted_centered':
                    old = json.loads((transport.PUBLIC/'groups'/(producer['trial']+'_source_fit_context_'+str(site)+'.json')).read_text())
                    if any(old[k] != v for k, v in hashes.items()) or old['diagnosis'] != diagnosis or old['selected_recordings'] != records:
                        raise ValueError('Frozen original SOURCE control differs')
                    old_exact = True
                groups.append(dict(site=str(site), diagnosis=diagnosis, selected_recordings=records,
                    old_control_exact=old_exact, **hashes))
            case = dict(trial=producer['trial'], family=family, checkpoint=ref['checkpoint'],
                rows=len(causal['ids']), groups=groups, registration_sha256=sha(REG),
                full_prediction_sha256=prior.frozen.fingerprint(prediction),
                result_source='fresh_run_SOURCE_frozen_Torch_cost_inference_cached_verified_models_and_inputs')
            if verify or path.exists():
                if case != json.loads(path.read_text()):
                    raise ValueError('Complete SOURCE cost replay differs')
            else:
                once(path, case)
            cases.append(case); beat(state='case_complete', completed=len(cases), total=2 if pilot else 18)
            fresh_cases += 1
            del state, prediction
        del data
    receipt = dict(pid=os.getpid(), seconds=time.monotonic()-started,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, registration_sha256=sha(REG),
        fresh_cases=fresh_cases, cached_verified_cases=cached_cases)
    refs = [artifact(PUBLIC/'cases'/(r['trial']+'_'+r['family']+'.json')) for r in cases]
    if pilot:
        if len(cases) != 2 or any(len(r['groups']) != 8 for r in cases):
            raise ValueError('Full first-producer resource pilot required')
        once(finish, dict(**receipt, engineering_pass=True, cases=refs,
            projected_full_seconds_not_completion=receipt['seconds']*9))
    else:
        summary = api.aggregate(cases, [r['trial'] for r in producers])
        if verify:
            if summary != json.loads((PUBLIC/'summary.json').read_text()):
                raise ValueError('Complete public summary replay differs')
            once(finish, dict(**receipt, all18_SOURCE_model_inferences_exact=True,
                all144_SOURCE_site_groups_exact=True, original72_source_control_groups_exact=True,
                summary_sha256=sha(PUBLIC/'summary.json')))
        else:
            once(PUBLIC/'summary.json', summary)
            once(finish, dict(**receipt, cases=refs, summary_sha256=sha(PUBLIC/'summary.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'pilot', 'run', 'verify'))
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    PUBLIC.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase == 'register':
            admission(True)
        else:
            execute(args.phase, args.resume)
