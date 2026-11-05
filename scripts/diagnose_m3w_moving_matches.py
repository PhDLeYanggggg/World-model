"""Attribute already-saved own-B matches without choosing new references."""
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_track_analogue as old
from src.evaluation import m3w_match_attribution as method
from src.world_model import m3w_kinematic_risk as kin
from src.world_model import m3w_rollout_history_risk as rel
from src.world_model import m3w_candidate_neighbor_basis as peer

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_match_attribution_v1'


def main():
    start = time.monotonic(); s, np = old.source, old.np
    a = s.read(old.PUBLIC/'analysis.json')
    old.rel.base.exact(s.sha(old.PUBLIC/'analysis.json'), s.read(old.PUBLIC/'verification.json')['analysis_sha256'])
    previous = s.read(old.previous.audit.PUBLIC/'analysis.json')
    records = []
    for context, ref in zip(a['contexts'], previous['contexts']):
        old.rel.base.exact(context['job'], ref['job']); data, _ = old.inputs(ref)
        for fold in context['folds']:
            saved = old.rel.base.read_checkpoint(s.checked(fold['checkpoint']))
            bank = old.restore_bank(saved['bank'], data)
            r = old.method.from_indices(bank, saved['result']['local_indices'], saved['result']['distances'])
            old.rel.base.exact(r, saved['result'])
            ix = data['sites'] == fold['source']; pos = r['donor_positions']
            z = np.clip((data['x']-bank['mean'])/bank['std'], -8, 8)
            record = method.summarize(z[ix], z[pos], data['weights'][pos], data['target'][ix], r['prediction'], data['weights'][ix])
            records.append(dict(job=context['job'], source=fold['source'], strata=record))
    result = dict(records=records, feature_names=list(kin.NAMES+rel.NAMES+peer.NAMES),
        result_source='fresh_run_descriptive_attribution_of_cached_verified_references',
        parent_analysis=s.artifact(old.PUBLIC/'analysis.json'),
        new_reference_search=False, model_selection=False, original_rules_unchanged=True,
        new_V_or_H=False, independent_confirmation=False, deployment_changed=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False)
    s.once(PUBLIC/'analysis.json', result)
    s.once(PUBLIC/'execution.json', dict(pid=s.os.getpid(), seconds=time.monotonic()-start,
        peak_RSS_bytes=s.resource.getrusage(s.resource.RUSAGE_SELF).ru_maxrss))


if __name__ == '__main__':
    main()
