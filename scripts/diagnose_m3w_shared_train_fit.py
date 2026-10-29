"""Frozen inner-fit census to distinguish fitting failure from recording shift."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from scripts import evaluate_m3w_shared_train_risk as evaluation
from scripts import report_m3w_shared_train_risk as verified

run, api = evaluation.run, evaluation.api
PUBLIC = run.PUBLIC/'fit_census'
REG = PUBLIC/'registration.json'


def summarize(rows, held):
    key = lambda row: (row['context'], row['source'], row['seed'])
    if len(rows) != 72 or len({key(r) for r in rows}) != 72 or set(map(key, rows)) != set(held):
        raise ValueError('All 72 fit/held pairs required')
    result = {}
    for arm in evaluation.ARMS:
        pairs = [(r['result']['policies'][arm], held[key(r)]['result']['policies'][arm]) for r in rows]
        score_pairs = [(r['result']['scores'][arm], held[key(r)]['result']['scores'][arm]) for r in rows]
        risks = lambda p: p['known_easy_positive_risk']
        ratios = [b/a for a, b in score_pairs if a is not None and b is not None and a > 0]
        result[arm] = dict(
            fit_known_easy_violations=sum(risks(a) is not None and risks(a) > .02+1e-12 for a, _ in pairs),
            held_known_easy_violations=sum(risks(b) is not None and risks(b) > .02+1e-12 for _, b in pairs),
            fit_known_easy_undefined=sum(risks(a) is None for a, _ in pairs),
            held_known_easy_undefined=sum(risks(b) is None for _, b in pairs),
            fit_known_pass_held_fail=sum(risks(a) is not None and risks(a) <= .02+1e-12
                and risks(b) is not None and risks(b) > .02+1e-12 for a, b in pairs),
            fit_completion_support=sum(a['finite_completion_supported'] for a, _ in pairs),
            held_completion_support=sum(b['finite_completion_supported'] for _, b in pairs),
            fit_mean_signed_MSE=statistics.mean(a for a, _ in score_pairs) if all(a is not None for a, _ in score_pairs) else None,
            held_mean_signed_MSE=statistics.mean(b for _, b in score_pairs) if all(b is not None for _, b in score_pairs) else None,
            median_held_fit_MSE_ratio=statistics.median(ratios) if ratios else None,
            defined_MSE_ratios=len(ratios))
    return dict(views=72, arms=result, rows_per_arm=sum(r['result']['rows'] for r in rows),
        unknown_rows_per_arm=sum(r['result']['unknown_rows'] for r in rows),
        optimizer_steps=0, new_held_inference=False, outer_development_read=False,
        independent_roles_read=False, deployment_changed=False, scientific_success=False)


def register():
    reg, cfg, parent = run.admission()
    complete = json.loads((evaluation.PUBLIC/'complete.json').read_text())
    if complete['summary_sha256'] != run.base.sha(evaluation.PUBLIC/'summary.json'):
        raise ValueError('Verified frozen held readout required')
    files = [Path(__file__).resolve(), run.ROOT/'tests/test_m3w_shared_train_fit_census.py', PUBLIC/'protocol.md']
    value = dict(bindings={str(p.relative_to(run.ROOT)): run.base.sha(p) for p in files},
        training_freeze_sha256=run.base.sha(run.PUBLIC/'training_freeze.json'),
        held_complete_sha256=run.base.sha(evaluation.PUBLIC/'complete.json'),
        fit_views=72, new_training=False, new_held_inference=False,
        outer_development_read=False, independent_roles_read=False)
    run.base.once(REG, value)
    return reg, cfg, parent


def execute(resume=False, check=False):
    reg, _, parent = register()
    for rel in [*json.loads(REG.read_text())['bindings'], str(REG.relative_to(run.ROOT))]:
        p = subprocess.run(['git', 'show', 'HEAD:'+rel], cwd=run.ROOT, capture_output=True)
        if p.returncode or p.stdout != (run.ROOT/rel).read_bytes():
            raise ValueError('Commit fixed census code before inference')
    held_complete = json.loads((evaluation.PUBLIC/'complete.json').read_text())
    held = {}
    for ref in held_complete['groups']:
        path = run.ROOT/ref['path']
        if run.base.sha(path) != ref['sha256']:
            raise ValueError('Held aggregate changed')
        row = json.loads(path.read_text())
        held[row['context'], row['source'], row['seed']] = row
    models = evaluation.frozen_models()
    if check:
        complete = json.loads((PUBLIC/'complete.json').read_text())
        rows = []
        for ref in complete['groups']:
            path = run.ROOT/ref['path']
            if (not path.resolve().is_relative_to((PUBLIC/'groups').resolve())
                    or path.is_symlink() or run.base.sha(path) != ref['sha256']):
                raise ValueError('Changed fit census group')
            row = json.loads(path.read_text())
            if row['registration_sha256'] != run.base.sha(REG) or not row['exact_inference_replay']:
                raise ValueError('Changed fit census provenance')
            verified.check_result(row['result'])
            item = next(i for i in reg['contexts'][row['context']] if i['packet']['identity']['source'] == row['source'])
            if set(row['result']['recordings']) != set(item['partition']['fit_recordings']):
                raise ValueError('Changed fit recording support')
            rows.append(row)
        if (complete['registration_sha256'] != run.base.sha(REG)
                or complete['summary_sha256'] != run.base.sha(PUBLIC/'summary.json')
                or summarize(rows, held) != json.loads((PUBLIC/'summary.json').read_text())):
            raise ValueError('Changed fit/held summary')
        print('Checked 72 frozen inner-fit census views; no new inference')
        return
    if (PUBLIC/'complete.json').exists():
        raise ValueError('Completed census immutable')
    api.torch.set_num_threads(4)
    api.torch.set_num_interop_threads(1)
    rows, refs = [], []
    started = time.monotonic()
    def beat(**kw):
        value = dict(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(), phase='fit_census', **kw)
        temp = run.PRIVATE/'fit_census_heartbeat.tmp'
        temp.write_text(json.dumps(value)+'\n')
        os.replace(temp, run.PRIVATE/'fit_census_heartbeat.json')
        print(json.dumps(value), flush=True)
    with (run.PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for context, items in sorted(reg['contexts'].items()):
            fit, _ = run.packets(items, parent, beat)
            for seed in reg['seeds']:
                shared = {arm: api.old.core.read_checkpoint(run.ROOT/models[context, seed, arm, None]['checkpoint']['path'])
                          for arm in evaluation.ARMS[1:]}
                for source, args in sorted(fit.items()):
                    path = PUBLIC/'groups'/(context+'_'+source+'_seed'+str(seed)+'.json')
                    if path.exists() and not resume:
                        raise ValueError('Explicit census resume required')
                    if path.exists():
                        row = json.loads(path.read_text())
                        if row['registration_sha256'] != run.base.sha(REG):
                            raise ValueError('Changed census registration')
                    else:
                        local = api.old.core.read_checkpoint(run.ROOT/models[context, seed, 'local', source]['checkpoint']['path'])
                        api.old.core.exact(local['preprocess'], args[-1])
                        predictions = {'local': api.old.predict(local, args[0], args[1])[0]}
                        api.np.testing.assert_array_equal(predictions['local'], api.old.predict(local, args[0], args[1])[0])
                        for arm, state in shared.items():
                            api.old.core.exact(state['source_preprocess'][source], args[-1])
                            predictions[arm] = api.predict_shared(state, source, args[0], args[1])
                            api.np.testing.assert_array_equal(predictions[arm], api.predict_shared(state, source, args[0], args[1]))
                        result, _ = evaluation.evaluate(args, predictions)
                        verified.check_result(result)
                        row = dict(context=context, source=source, seed=seed, result=result,
                            registration_sha256=run.base.sha(REG), exact_inference_replay=True,
                            result_source='fresh_run_frozen_inner_fit_inference', new_training=False,
                            new_held_inference=False, outer_development_read=False,
                            independent_roles_read=False, deployment_changed=False)
                        run.base.once(path, row)
                    rows.append(row)
                    refs.append(dict(path=str(path.relative_to(run.ROOT)), sha256=run.base.sha(path)))
                    beat(state='view_complete', views=len(rows), total=72)
        run.base.once(PUBLIC/'summary.json', summarize(rows, held))
        run.base.once(PUBLIC/'complete.json', dict(groups=refs, seconds=time.monotonic()-started,
            registration_sha256=run.base.sha(REG), summary_sha256=run.base.sha(PUBLIC/'summary.json'),
            new_training=False, new_held_inference=False, independent_roles_read=False))
        beat(state='complete', views=72)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'run', 'check'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.action == 'register':
        register()
        print('Registered frozen inner-fit census; no held refit or threshold search')
    else:
        execute(resume=args.resume, check=args.action == 'check')
