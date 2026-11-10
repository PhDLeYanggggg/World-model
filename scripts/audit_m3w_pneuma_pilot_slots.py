"""Count matched edit feasibility without ranking or evaluating forecasts."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_pneuma_prefix_pilot as pilot


bank, storage, np = pilot.transfer.bank, pilot.storage, pilot.np


def main():
    inp, pred = (pilot.OUT/k for k in ('prefix_receipt.json', 'forecast_receipt.json'))
    a, b = pilot.read(inp), pilot.read(pred)
    if (b['binding']['prefix_receipt'] != storage.sha(inp) or b['forecasts_scored'] or b['future_coordinates_loaded']
            or b['predictive_roles_assigned'] or a['predictive_roles_assigned'] or set(a['files']) != set(b['files'])):
        raise ValueError('Need split-neutral, unscored forecasts from verified prefixes')
    seeds = b['neural_seeds']
    tables, stats, largest = {}, {seed: [] for seed in seeds}, 0
    for name, f in b['files'].items():
        stem = Path(name).stem
        p, q = (pilot.LOCAL/f'{stem}.{s}.npz' for s in ('prefix', 'forecast'))
        if (storage.sha(p) != f['prefix_sha256'] or storage.sha(q) != f['forecast_sha256']
                or f['prefix_sha256'] != a['files'][name]['prefix_sha256']):
            raise ValueError('Forecast or observed input changed')
        with np.load(p, allow_pickle=False) as x, np.load(q, allow_pickle=False) as y:
            offsets = x['offsets']
            if len(offsets)-1 != f['queries'] or offsets[-1] != f['agents']:
                raise ValueError('Changed roster')
            counts = {seed: bank.count_queries(y, offsets, seed) for seed in seeds}
        for seed in seeds:
            stats[seed].append(counts[seed])
        n = int(np.diff(offsets).max(initial=0))
        largest = max(n, largest)
        tables[name] = dict(queries=f['queries'], agents=f['agents'], maximum_agents=n, by_seed={
            str(s): bank.aggregate(v) for s, v in counts.items()})
    total = {str(s): bank.aggregate(np.concatenate(v)) for s, v in stats.items()}
    if any(v['queries'] != a['queries'] for v in total.values()):
        raise ValueError('Lost queries in structural budget audit')
    result = dict(result_source='fresh_label_free_slot_count', prefix_receipt=storage.sha(inp),
        forecast_receipt=storage.sha(pred), slot_counter_sha256=storage.sha(Path(bank.__file__)),
        relative_radius_fraction=bank.FRACTION, total_queries=a['queries'], maximum_agents=largest,
        per_file=tables, totals=total, ranked=False, forecasts_scored=False, predictive_roles_assigned=False,
        future_coordinates_loaded=False, formal_clock_selected=False, original_2pct=True)
    storage.write_json(pilot.OUT/'slot_receipt.json', result)
    print('Queries:', a['queries'], 'maximum roster:', largest, 'slots:', total, flush=True)


if __name__ == '__main__':
    main()
