"""SOURCE-only current-bank feasibility; no development or calibration access."""
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 before runtime imports')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'4')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_selection_cardinal_cost as parent
from scripts import run_m3w_selection_cost_transport as transport
from src.evaluation.m3w_selection_bank_support import diagnose

NAME = 'european_selection_bank_support_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME


def main():
    _,producers,ids = parent.admission(); start = time.monotonic()
    if (PUBLIC/'complete.json').exists(): raise ValueError('Completed support diagnostic immutable')
    parent.prior.api.torch.set_num_threads(4); parent.prior.api.torch.set_num_interop_threads(1)
    source = parent.prior.source_inputs(json.loads(parent.prior.frozen.REG.read_text())); rows = []
    for row in producers:
        case = transport.source_case(row,source,ids); c = case['causal']
        result = diagnose(case['target'],c['allowed'],case['recordings'])
        rows.append(dict(trial=row['trial'],producer=row['checkpoint'],
            source_sites=sorted(set(c['sites'])),ids_sha256=parent.prior.frozen.fingerprint(c['ids']),
            native_cost_labels_sha256=parent.prior.frozen.fingerprint(case['target']),diagnosis=result))
        heartbeat = dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                         state='SOURCE_bank_support',complete_producers=len(rows),total=9)
        parent.write(PRIVATE/'heartbeat.json',heartbeat); print(json.dumps(heartbeat),flush=True)
        del case,c
    if len(rows) != 9: raise ValueError('All original SOURCE producer scopes required')
    value = dict(result_source='fresh_SOURCE_native_cost_support_arithmetic_cached_verified_forecast_banks',
        pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        rows=rows,required_record_views=sum(r['diagnosis']['required_record_views'] for r in rows),
        bank_infeasible_record_views=sum(r['diagnosis']['bank_infeasible_record_views'] for r in rows),
        code_sha256=parent.sha(ROOT/'src/evaluation/m3w_selection_bank_support.py'),
        plan_sha256=parent.sha(PUBLIC/'plan.md'),new_training=False,policy_or_risk_changed=False,
        development_labels_read=False,calibration_or_confirmation_access=False,oracle_choices_exported=False,
        SOURCE_fitting_diagnostic_not_generalization=True,stage5c_executed=False,smc_enabled=False)
    parent.once(PUBLIC/'complete.json',value)
    print(json.dumps({k:value[k] for k in ('pid','seconds','required_record_views','bank_infeasible_record_views')}))


if __name__ == '__main__':
    PUBLIC.mkdir(parents=True,exist_ok=True); PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); main()
