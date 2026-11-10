"""Prepare equal-budget forecast proposals for the unscored Swiss pilot."""

from collections import Counter
import fcntl
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_pneuma_matched_pilot as matched
from scripts import build_m3w_swiss_roundabout_pilot as pilot

source, np, torch = pilot.source, matched.np, matched.torch
LOCAL = pilot.LOCAL/'matched'


def run():
    p = source.OUT/'forecast_receipt.json'
    x = source.OUT/'prefix_receipt.json'
    pre, frozen = pilot.read(x), pilot.read(p)
    if (frozen['binding']['prefix'] != source.digest(x) or frozen['roles_assigned'] or frozen['forecasts_scored']
            or frozen['future_coordinates_parsed'] or frozen['training_updates'] or len(frozen['files']) != 18):
        raise ValueError('Require the verified unscored Swiss bank')
    binding = dict(forecast=source.digest(p), prefix=source.digest(x),
                   driver=source.digest(Path(__file__)), search=source.digest(Path(matched.__file__)),
                   pair_search=source.digest(Path(matched.work.__file__)))
    results = {}
    for seed in frozen['seeds']:
        net, head = matched.work.heads(seed)
        for name, item in frozen['files'].items():
            stem = Path(name).stem
            hp, fp = (pilot.LOCAL/f'{stem}.{k}.npz' for k in ('prefix', 'forecast'))
            if source.digest(hp) != pre['files'][name]['prefix_sha256'] or source.digest(fp) != item['forecast_sha256']:
                raise ValueError('Changed prefix or forecast file')
            tag = f'{stem}_{seed}'
            packet, rec = (LOCAL/f'{tag}.{k}' for k in ('npz', 'json'))
            ident = dict(binding=binding, head_freeze=head, seed=seed, member=name,
                         forecast_sha256=item['forecast_sha256'], prefix_sha256=item['prefix_sha256'])
            if not rec.exists():
                with np.load(hp, allow_pickle=False) as past, np.load(fp, allow_pickle=False) as pred:
                    offsets = past['offsets']
                    counts = matched.prior.bank.bank.count_queries(pred, offsets, seed)
                    eligible = np.flatnonzero((counts[:, 1] > 0.) & (counts[:, 2] > 0.))
                    if len(eligible):
                        take = np.concatenate([np.arange(offsets[i], offsets[i+1]) for i in eligible])
                        new = np.r_[0, np.cumsum(np.diff(offsets)[eligible])]
                        a = matched.proposals(past['history'][take], new, {k: pred[k][take] for k in pred.files},
                                              seed, net, lambda _: None)
                        for k in matched.ARMS:
                            a[f'{k}_rows'] = take[a[f'{k}_rows']]
                    else:
                        a = {}
                different = {}
                if len(eligible):
                    chosen = a['joint_rows']*3+a['joint_families']
                    different = {k: int(np.any(chosen != a[f'{k}_rows']*3+a[f'{k}_families'], axis=1).sum())
                                 for k in matched.ARMS[:-1]}
                digest = pilot.save(packet, dict(a, query_index=eligible))
                source.write(rec, dict(identity=ident, sha256=digest, available=len(eligible), original=len(offsets)-1,
                                       different_from_joint=different))
            row = pilot.read(rec)
            if row['identity'] != ident or row['sha256'] != source.digest(packet) or row['original'] != item['queries']:
                raise ValueError('Changed matched proposal packet')
            results[tag] = row
            pilot.beat('proposal', f'{tag}', len(results) % 18 or 18)
    if len(results) != 54:
        raise ValueError('Incomplete three-seed proposal bank')
    differences = sum((Counter(r['different_from_joint']) for r in results.values()), Counter())
    source.write(source.OUT/'matched_receipt.json', dict(result_source='fresh_unscored_equal_budget_forecasts',
        binding=binding, packets=results, original_views=sum(r['original'] for r in results.values()),
        comparable_views=sum(r['available'] for r in results.values()), edits_per_arm=2,
        equal_peak_radius=True, distinct_recipients=True, different_from_joint=differences,
        eligibility='two distinct recipients and a legal same-family pair in the causal forecast bank',
        future_coordinates_parsed=False, roles_assigned=False, forecasts_scored=False))


if __name__ == '__main__':
    if sys.platform == 'darwin' and platform.machine() != 'arm64':
        raise SystemExit('Use the native arm64 .venv-pytorch interpreter')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
