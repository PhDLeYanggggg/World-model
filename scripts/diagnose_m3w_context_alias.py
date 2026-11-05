"""Measure two-agent pooling alias support on own-B, never held outcomes."""
import argparse
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import diagnose_m3w_discrimination as parent
from src.evaluation.m3w_decision_semantics import joint_target

source, api = parent.source, parent.api


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true'); args = parser.parse_args()
    public, private = parent.PUBLIC, parent.PRIVATE
    with (private/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        parent.parent.old.previous.teacher.lineage.PRIVATE = private
        started, guard, beat = parent.parent.old.previous.teacher.lineage.runtime({},
            source.read(parent.parent.CONFIG), 'context_alias')
        _, values = parent.parent.register(beat)
        source.once(public/'context_registration.json', dict(script=source.artifact(Path(__file__)),
            own_B_only=True, posthoc_diagnostic=True, no_new_V_or_H=True))
        rows = []
        for job in values[3]:
            data, warm = parent.parent.inputs(values, job); c = data['causal']
            y = joint_target(api.parent.kin.parent.signed(data['truth_by_arm']['bound_aware']).numpy())
            sizes = source.np.array([len(g) for g in c['groups']])
            grouped, conflicting = [], []
            for g in c['groups']:
                if len(g) != 2:
                    continue
                legal = c['allowed'][g] & data['observed_future'][g, None]
                shared = legal.all(0)
                grouped.extend((int(a), int(g[0]), int(g[1])) for a in source.np.flatnonzero(shared))
                conflicting.extend((int(a), int(g[0]), int(g[1])) for a in source.np.flatnonzero(
                    shared & (((y[g[0]] > 0) & (y[g[1]] < 0)) | ((y[g[1]] > 0) & (y[g[0]] < 0)))))
            rows.append(dict(job=job['key'], rows=len(c['ids']), queries=len(sizes),
                group_size_counts={str(int(s)): int((sizes == s).sum()) for s in source.np.unique(sizes)},
                two_agent_queries=int((sizes == 2).sum()), two_agent_same_proposal_observed_pairs=len(grouped),
                two_agent_opposite_sign_pairs=len(conflicting),
                mean_relative_self_coefficient=float(source.np.mean(source.np.where(sizes > 1,
                    (sizes-2)/source.np.maximum(sizes-1, 1), 1))),
                note='Self plus mean(other) aliases two agents for a common action; count is fitting support, not proof of the dominant failure cause'))
            del data, warm, y; source.gc.collect(); guard()
        path = public/'context_alias.json'
        source.once(path, dict(status='fresh_run_own_B_only', rows=rows))
        if args.verify:
            source.once(public/'context_verification.json', dict(exact=True, sha256=source.sha(path)))
        source.once(public/('context_verify_execution.json' if args.verify else 'context_execution.json'),
            dict(pid=source.os.getpid(), seconds=time.monotonic()-started))
        beat(state='complete')


if __name__ == '__main__':
    main()
