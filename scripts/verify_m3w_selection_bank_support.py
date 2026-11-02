"""Independent SOURCE support reconstruction and causal-envelope mechanism check."""
import fcntl
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import diagnose_m3w_selection_bank_support as run

parent,prior = run.parent,run.parent.prior
np = prior.np


def recording_views(records, saved):
    native = sorted(set(records))
    if len({str(rec) for rec in native}) != len(native) or {str(rec) for rec in native} != set(saved):
        raise ValueError('Native and serialized recording rosters differ')
    for rec in native:
        yield records == rec, saved[str(rec)]


def main():
    _,producers,ids = parent.admission(); start = time.monotonic()
    finish = run.PUBLIC/'verification.json'
    if finish.exists(): raise ValueError('Complete SOURCE verification immutable')
    original = json.loads((run.PUBLIC/'complete.json').read_text())
    if original['code_sha256'] != parent.sha(ROOT/'src/evaluation/m3w_selection_bank_support.py'):
        raise ValueError('Original support arithmetic changed')
    prior.api.torch.set_num_threads(4); prior.api.torch.set_num_interop_threads(1)
    source = prior.source_inputs(json.loads(prior.frozen.REG.read_text())); models = parent.fitted()
    saved = {r['trial']:r for r in original['rows']}; cases = []; amplitudes = []; checked_records = 0
    for row in producers:
        case = run.transport.source_case(row,source,ids); c,y = case['causal'],case['target']
        old = saved[row['trial']]
        if (prior.frozen.fingerprint(c['ids']) != old['ids_sha256']
                or prior.frozen.fingerprint(y) != old['native_cost_labels_sha256']
                or set(c['sites']) & set(row['fit_sites']) or len(set(c['sites'])) != 8):
            raise ValueError('Original producer-excluded SOURCE rows/costs required')
        known = np.isfinite(y).all((1,2))
        eligible = c['allowed'] & known[:,None] & (y[:,:,3] > 0)
        residual = np.where(eligible,np.maximum(y[:,:,1]-.02*y[:,:,2],y[:,:,4]-.02*y[:,:,3]),np.inf)
        safe = residual <= 0
        for at,d in recording_views(case['recordings'],old['diagnosis']['recordings']):
            actual = dict(rows=int(at.sum()),known_rows=int((at&known).sum()),
                wholly_unknown_rows=int((at&~known).sum()),full_known_easy_reference_mass=float(y[at&known,0,3].sum()),
                legal_known_easy_action_occurrences=int(eligible[at].sum()),
                safe_known_easy_action_occurrences=int(np.count_nonzero(residual[at] <= 0)),
                strictly_beneficial_safe_occurrences=int(np.count_nonzero(safe[at]&(y[at,:,0] > 0))),
                finite_support_bank_feasible=bool(np.min(residual[at]) <= 0))
            if any(actual[k] != d[k] for k in actual): raise ValueError('Independent native support reconstruction differs')
            checked_records += 1
        ref = json.loads((prior.PRIVATE/'source_banks'/row['trial']/'complete.json').read_text())
        neural = np.load(parent.checked(ref['prediction']),mmap_mode='r',allow_pickle=False)
        origin = source['origin'][c['ids']]
        physical = np.stack([prior.baseline_numpy(source['history'][c['ids']],k)-origin[:,None] for k in range(6)],1)
        paths = np.concatenate((physical,neural[:,None].astype(float)),1)
        tight = np.linalg.norm(paths-paths[:,row['baseline_index'],None],axis=-1).max(-1)
        legacy = np.maximum(tight,np.linalg.norm(paths-paths[:,1,None],axis=-1).max(-1))
        np.testing.assert_array_equal(legacy,c['envelope']); np.testing.assert_array_equal(tight > 0,c['allowed'])
        native_overrun = (y[:,:,0]+y[:,:,1]-tight)[known]
        if np.any(native_overrun > 1e-8*np.maximum(1,tight[known])):
            raise ValueError('Native benefit plus harm violates actual-floor disagreement')
        p = prior.api.predict(prior.api.read_checkpoint(parent.checked(models[row['trial'],'mean_absolute']['checkpoint'])),c)
        source_control = json.loads((ROOT/'outputs/publication_readiness_2026_09/european_calibration_source_fit_v1/cases'/
                                     (row['trial']+'_mean_absolute.json')).read_text())
        if prior.frozen.fingerprint(p) != source_control['full_prediction_sha256']:
            raise ValueError('Original mean-cost SOURCE inference changed')
        choice,_ = prior.allocator.select_actions(p,c['allowed'],arm='physical_neural')
        selected = np.zeros_like(c['allowed']); at = np.flatnonzero(choice >= 0); selected[at,choice[at]] = True
        overrun = p[:,:,0]+p[:,:,1]-tight
        robust = overrun > 5e-6*np.maximum(1,tight)
        scopes = {}
        neural_only = np.zeros_like(c['allowed']); neural_only[:,6] = c['allowed'][:,6]
        for scope,take in (('all_legal',c['allowed']),('neural_legal',neural_only),('old_selected',selected)):
            ratios = legacy[take]/tight[take]
            scopes[scope] = dict(actions=int(take.sum()),
                legacy_to_actual_radius_quantiles=np.quantile(ratios,[0,.5,.9,.99,1]).tolist() if len(ratios) else None,
                strict_cost_sum_overrun_actions=int((take&(overrun > 0)).sum()),
                cost_sum_overrun_beyond_5ppm_actions=int((take&robust).sum()),
                source_true_cost_max_positive_overrun=float(max(0.,native_overrun.max())))
        for rec in sorted(set(case['recordings'])):
            take = safe & (case['recordings'] == rec)[:,None]
            benefit = take&(y[:,:,0] > 0)
            amplitudes.append(dict(trial=row['trial'],recording=str(rec),
                max_safe_radius_pixel=float(tight[take].max()) if take.any() else None,
                max_beneficial_safe_radius_pixel=float(tight[benefit].max()) if benefit.any() else None,
                max_safe_benefit_pixel=float(y[:,:,0][benefit].max()) if benefit.any() else None,
                beneficial_safe_actions_radius_ge_point1_pixel=int((benefit&(tight >= .1)).sum())))
        cases.append(dict(trial=row['trial'],baseline_index=row['baseline_index'],scopes=scopes,
            mean_cost_SOURCE_prediction_exact=True,actual_radius_reconstructs_causal_availability=True))
        beat = dict(pid=os.getpid(),state='SOURCE_support_and_radius_verified',producers=len(cases),total=9)
        parent.write(run.PRIVATE/'verification_heartbeat.json',beat); print(json.dumps(beat),flush=True)
        del case,c,y,p,paths,physical,neural
    if checked_records != original['required_record_views'] or len(cases) != 9:
        raise ValueError('All444record views and nine SOURCE producers required')
    value = dict(result_source='fresh_SOURCE_replay_independent_risk_residual_reconstruction_and_causal_geometry',
        pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        all_record_support_counts_exact=True,required_record_views=checked_records,
        all9_old_mean_cost_inferences_exact=True,native_costs_respect_actual_floor_radius=True,
        cases=cases,source_support_amplitudes=amplitudes,
        complete_sha256=parent.sha(run.PUBLIC/'complete.json'),script_sha256=parent.sha(Path(__file__)),
        tolerance_role='5ppm_descriptive_numerical_sensitivity_not_new_policy_threshold',
        new_training=False,policy_or_risk_changed=False,development_labels_read=False,
        calibration_or_confirmation_access=False,oracle_choices_exported=False,
        SOURCE_fitting_diagnostic_not_generalization=True,stage5c_executed=False,smc_enabled=False)
    parent.once(finish,value)
    print(json.dumps(dict(pid=os.getpid(),seconds=value['seconds'],records=checked_records,complete=True)))


if __name__ == '__main__':
    with (run.PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); main()
