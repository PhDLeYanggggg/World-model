"""Label-aware SOURCE inner-held support ceiling; never an inference policy."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_multiaction_cost as run
from src.evaluation.m3w_multiaction_support import solve_action_bank


def audit(verify=False):
    cfg, reg, _, data, jobs, identity = run.admission()
    started, beat, guard = run.runtime(cfg)
    prior = json.loads((run.previous.PUBLIC/'multiaction_support.json').read_text())
    fitting = {(r['context'], r['source'], r['recording']):r for r in prior['rows']}
    rows = []
    for context, items in sorted(reg['contexts'].items()):
        scopes = run.prepare_context(context, items, data, jobs, identity, beat, include_held=True)
        for scope, packets in scopes.items():
            for source, packet in sorted(packets.items()):
                for recording in sorted(set(packet['recordings'])):
                    take = packet['recordings'] == recording
                    banks = {}
                    for arm, indices in run.api.ARMS.items():
                        indices = list(indices)
                        banks[arm], choices = solve_action_bank(packet['y'][take][:, indices],
                            packet['envelope'][take][:, indices], packet['allowed'][take][:, indices])
                        if scope == 'fit':
                            key = dict(physical_only='physical_only', physical_neural='physical_plus_original',
                                       neural_only='original_neural_real_only')[arm]
                            if banks[arm] != fitting[context, source, str(recording)]['banks'][key]:
                                raise ValueError('Original148-view fitting oracle does not reproduce')
                    known = run.np.isfinite(packet['y'][:, 0]).all(1)
                    easy = known & (packet['y'][:, 0, 3] > 0)
                    eligible = packet['allowed'].any(1)
                    rows.append(dict(context=context, source=source, recording=str(recording), scope=scope,
                        rows=int(take.sum()), known_rows=int((take & known).sum()),
                        known_easy_rows=int((take & easy).sum()),
                        known_eligible_easy_rows=int((take & easy & eligible).sum()), banks=banks))
        beat(state='support_ceiling_context_complete', context=context, record_views=len(rows))
        del scopes
    summary = {}
    for scope in ('fit', 'inner_held'):
        subset = [r for r in rows if r['scope'] == scope]
        summary[scope] = dict(required_record_context_views=len(subset),
            unique_source_recordings=len({(r['source'], r['recording']) for r in subset}),
            banks={arm:dict(budget_feasible=sum(r['banks'][arm]['budget_feasible'] for r in subset),
                positive_feasible=sum(r['banks'][arm]['positive_feasible'] for r in subset),
                blockers=[dict(context=r['context'], source=r['source'], recording=r['recording'],
                    reason=r['banks'][arm]['infeasibility_reason'], known_easy_rows=r['known_easy_rows'],
                    known_eligible_easy_rows=r['known_eligible_easy_rows'])
                    for r in subset if not r['banks'][arm]['budget_feasible']]) for arm in run.api.ARMS})
    value = dict(result_source='fresh_run_label_aware_source_inner_held_support_diagnostic',
        summary=summary, rows=rows, code_sha256=run.sha(Path(__file__)),
        original_fitting_oracle_views_replayed=148, head_seeds_not_repeated=True,
        future_labels_used_only_by_offline_oracle=True, independent_roles_opened=False,
        new_training=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    path = run.PUBLIC/'inner_held_support_diagnostic.json'
    if verify:
        if value != json.loads(path.read_text()):
            raise ValueError('Exact SOURCE inner-held support ceiling replay differs')
    else:
        run.once(path, value)
    run.json_write(run.PRIVATE/('inner_held_support_verify_runtime.json' if verify else 'inner_held_support_runtime.json'),
                   dict(pid=run.os.getpid(), seconds=time.monotonic()-started, exact_replay=verify))
    print(json.dumps(dict(exact_replay=verify, summary=summary,
                          result_sha256=run.sha(path))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true'); args = parser.parse_args()
    run.PRIVATE.mkdir(parents=True, exist_ok=True)
    with (run.PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        audit(args.verify)
