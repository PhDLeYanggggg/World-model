"""Exercise the Songdo scorer on real frozen proposals and fabricated targets only."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_songdo_matched_proposals as proposals
from src.evaluation import m3w_songdo_matched_readout as scorer

np, pack, first = proposals.np, proposals.pack, proposals.first
OUT = first.PUBLIC.parent / 'songdo_readout_contract_v1'


def make_fixture(cv, median):
    # Arbitrary, deterministic perturbations and missing labels, not a forecast benchmark.
    actor = np.arange(len(cv))
    target = cv+.6*(median-cv)
    target[:, :, 0] += np.where(actor%2 == 0, .1, -.1)[:, None]
    valid = np.ones(cv.shape[:2], dtype=bool)
    valid[actor%9 == 0] = False
    valid[actor%5 == 0, -4:] = False
    return np.where(valid[..., None], target, np.nan), valid


def proposal_for_seed(packet, seed):
    k = f'seed{seed}_'
    if k+'bounds' not in packet:
        return None
    return dict(bounds=packet[k+'bounds'], radius=float(packet[k+'radius'].item()),
                arms={a: (packet[k+a+'_rows'], packet[k+a+'_paths']) for a in scorer.ARMS})


def run():
    q = proposals.OUT / 'receipt.json'
    b = first.PUBLIC / 'vehicle_bank.json'
    g = first.PUBLIC / 'lineage.json'
    frozen, bank, lineage = (first.read(p) for p in (q, b, g))
    if (frozen['binding']['candidate_bank'] != pack.sha(b)
            or frozen['binding']['lineage'] != pack.sha(g)
            or any(frozen[k] for k in ('split_assigned', 'future_index_loaded', 'forecasts_scored'))
            or set(frozen['sessions']) != set(bank['sessions'])):
        raise ValueError('Require the frozen unscored sources and proposals')
    groups = {s: '+'.join(c) for c in lineage['candidate_source_components'] for s in c}
    views, flight_components = {}, {}
    count = dict(flights=0, seed_views=0, matched_seed_views=0, roster_rows=0, partial_mask_rows=0,
                 wholly_masked_rows=0, full_roster_windows=0, pair_supported_windows=0)
    for name, item in frozen['sessions'].items():
        stem = Path(name).stem
        a, d = proposals.bank.CACHE / f'{stem}.npz', proposals.LOCAL / f'{stem}.npz'
        if (pack.sha(a), pack.sha(d)) != (bank['sessions'][name]['sha256'], item['packet_sha256']):
            raise ValueError(f'Frozen observed packet changed: {name}')
        with np.load(a, allow_pickle=False) as v, np.load(d, allow_pickle=False) as z:
            cv, ols, offsets = v['cv'], v['ols4'], v['offsets']
            y, valid = make_fixture(cv, v['median4'])
            views[name] = {}
            for seed in scorer.SEEDS:
                option = proposal_for_seed(z, seed)
                s = scorer.score_flight(cv, ols, y, valid, offsets, option, easy_cut=.25, hard_cut=1.)
                if (option is not None) != s['matched_eligible'] or s['matched_eligible'] != (
                        item['seeds'][str(seed)] is not None):
                    raise ValueError('Matched roster mismatch')
                count['seed_views'] += 1
                count['matched_seed_views'] += int(s['matched_eligible'])
                views[name][seed] = dict(matched_contrasts=s['matched_contrasts'])
            count['flights'] += 1
            count['roster_rows'] += len(y)
            count['partial_mask_rows'] += int((valid.any(1) & ~valid.all(1)).sum())
            count['wholly_masked_rows'] += int((~valid.any(1)).sum())
            count['full_roster_windows'] += s['coverage']['complete_roster_windows']
            count['pair_supported_windows'] += s['coverage']['complete_pair_windows']
        flight_components[name] = groups[item['site']]
    summaries = {}
    for arm in scorer.ARMS[:-1]:
        d = scorer.component_interval(views, flight_components, arm, 'lower_pp')
        if d['status'] != 'estimable' or len(d['by_component']) != len(set(groups.values())):
            raise ValueError('The fabricated fixture should cover every source component')
        summaries[arm] = d['flights_by_component']
    OUT.mkdir(parents=True, exist_ok=True)
    pack.write_json(OUT / 'receipt.json', dict(result_source='synthetic_targets_on_frozen_real_proposals',
        bindings=dict(proposals=pack.sha(q), vehicle_bank=pack.sha(b), lineage=pack.sha(g),
                      scorer=pack.sha(Path(scorer.__file__)), verifier=pack.sha(Path(__file__))),
        counts=count, source_components=len(set(groups.values())), matched_flights_by_component=summaries,
        pseudo_effect_sizes_reported=False, real_future_index_loaded=False, real_future_coordinates_loaded=False,
        real_forecasts_scored=False, split_assigned=False, calibration_fitted=False))
    print(count)


if __name__ == '__main__':
    run()
