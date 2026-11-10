"""Check full-grid easy support and risk on the archived Songdo proposals."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_m3w_songdo_full_grid_risk import ARMS, OUT as prior_out, np, primary, prior
from src.evaluation import m3w_full_grid_risk as risk_math


OUT = prior_out.with_name('songdo_full_grid_easy_risk_v1')
LIMIT = .02


def read_flight(data, name, archived, easy_cut):
    meta, cache, prefix, bank = data.observed(name)
    packet = primary.checked(primary.PACKETS/f'{Path(name).stem}.npz', data.proposals[name]['packet_sha256'])
    with (np.load(prefix, allow_pickle=False) as p, np.load(bank, allow_pickle=False) as b,
          np.load(packet, allow_pickle=False) as z):
        target, valid = data.targets(name, cache, p, meta)
        np.testing.assert_array_equal(p['offsets'], b['offsets'])
        cv = b['cv']
        for seed in primary.protocol.SEEDS:
            old = archived['scores'][str(seed)]
            for arm in ARMS:
                key = f'seed{seed}_{arm}'
                rows, path = z[key+'_rows'], z[key+'_paths']
                args = (cv[rows], path, target[rows], valid[rows])
                all_risk, easy = risk_math.assess(*args), risk_math.assess_easy(*args, easy_cut=easy_cut, limit=LIMIT)
                risk = old['matched']['arms'][arm]['risk']
                original = risk['completion_easy_risk_upper']
                replay = easy['original_easy']
                if (replay is None) != (original is None):
                    raise ValueError('Original selected-easy support changed')
                if original is not None:
                    np.testing.assert_allclose(replay, original, atol=1e-12, rtol=0.)
                    signed = risk['known_easy_net_degradation']
                    if signed is None or signed > original+1e-12:
                        raise ValueError('Signed easy harm must be dominated by positive selected-easy harm')
                full = all_risk['full_grid_upper']
                yield arm, old['released'][arm]['edited'], dict(
                    original_pass=risk['supported'], all_pass=full is not None and full <= LIMIT,
                    easy_support=easy['easy_support_guaranteed'],
                    easy_violation=easy['easy_excess_supremum'] > 0., easy_pass=easy['selected_easy_pass_for_all'])


def add(c, v):
    c.update(views=1, original_three=int(v['original_pass']), full_grid_all=int(v['all_pass']),
             full_grid_easy_support=int(v['easy_support']), full_grid_easy=int(v['easy_pass']),
             full_grid_all_and_easy=int(v['all_pass'] and v['easy_pass']))
    if v['original_pass'] and v['all_pass']:
        c.update(original_three_and_full_grid_all=1,
                 also_full_grid_easy=int(v['easy_pass']),
                 possible_undefined_easy=int(not v['easy_support']),
                 possible_easy_violation=int(v['easy_violation']))


def run():
    data, names, hashes, binding, original = prior.check_scope()
    previous = prior.read(prior_out/'readout.json')
    cut_path = prior.published.PUBLIC/'calibration_receipt.json'
    cut = prior.read(cut_path)
    if (prior.storage.sha(cut_path) != original['binding']['published_calibration_receipt_sha256']
            or cut['freeze_sha256'] != binding['release_freeze_sha256']
            or cut['calibration_cutoffs_sha256'] != original['binding']['calibration_cutoffs_sha256']
            or any(previous['binding'][k] != binding[k] for k in ('flight_score_hash_root', 'primary_score_sha256'))):
        raise ValueError('Cutoff, outcomes or previous sensitivity has changed')
    table = {arm: dict(matched=Counter(), released=Counter()) for arm in ARMS}
    for i, name in enumerate(sorted(names), 1):
        old = prior.verified_flight(prior.published.RUN/'confirmation'/f'{Path(name).stem}.json', hashes[name])
        for arm, edited, result in read_flight(data, name, old, cut['easy_cut']):
            add(table[arm]['matched'], result)
            if edited:
                add(table[arm]['released'], result)
        if i % 20 == 0:
            print(f'full-grid easy: {i}/160', flush=True)
    for arm, counts in table.items():
        for k, v in counts.items():
            before = previous['results'][arm]['matched' if k == 'matched' else 'actual_releases']
            for present, old_key in [('views', 'total'), ('original_three', 'original_three_checks'),
                                     ('full_grid_all', 'full_grid_all_pass'),
                                     ('original_three_and_full_grid_all', 'original_three_and_full_grid_all')]:
                if v[present] != before.get(old_key, 0):
                    raise ValueError('Prior all-edited table failed to replay')
    result = dict(result_source='fresh_retrospective_full_grid_easy_risk_sensitivity',
                  binding=dict(binding, all_risk_readout_sha256=prior.storage.sha(prior_out/'readout.json'),
                               full_grid_easy_driver_sha256=prior.storage.sha(Path(__file__)),
                               full_grid_risk_code_sha256=prior.storage.sha(Path(risk_math.__file__)),
                               full_grid_easy_protocol_sha256=prior.storage.sha(OUT/'protocol.md')),
                  flights=len(names), components=len(set(names.values())), seeds=primary.protocol.SEEDS,
                  easy_cut=cut['easy_cut'], risk_limit=LIMIT, results=table,
                  original_2pct_unchanged=True, original_scored_outcomes_reused=True,
                  new_external_cohort_opened=False, calibration_outcomes_read=False,
                  deployment_changed=False, fit=False, submission_ready=False)
    prior.storage.write_json(OUT/'readout.json', result)
    print(table, flush=True)


if __name__ == '__main__':
    run()
