"""Compare sampled and policy-selected path families without using outcomes."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import fit_m3w_vehicle_nested_nominees as nested


ctx, screen, action, np, torch, base = nested.ctx, nested.screen, nested.action, nested.np, nested.torch, nested.base
read, write, sha, old = nested.read, nested.write, nested.sha, nested.old
OUT = nested.OUT


def compare(bank, legal, fixed):
    if bank.ndim != 5 or fixed.shape != (*bank.shape[:2], *bank.shape[-2:]) or legal.shape != bank.shape[:3]:
        raise ValueError('Require two aligned recipient paths per query')
    if not legal.any(axis=2).all():
        raise ValueError('Frozen recipient has no legal family')
    match = legal & (np.abs(bank-fixed[:, :, None]).max(axis=(-1, -2)) <= 1e-10)
    if not match.any(axis=2).all():
        raise ValueError('Frozen action does not match any archived family')
    actual = match/match.sum(axis=2, keepdims=True)
    uniform = legal/legal.sum(axis=2, keepdims=True)
    return dict(actors=int(legal.shape[0]*legal.shape[1]), views=int(len(legal)),
        ambiguous=int((match.sum(axis=2) > 1).sum()), actual=actual.sum((0, 1)).tolist(),
        uniform=uniform.sum((0, 1)).tolist(), actual_same=float((actual[:, 0]*actual[:, 1]).sum()),
        uniform_same=float((uniform[:, 0]*uniform[:, 1]).sum()))


def main():
    source, forecasts, parent, _, _ = ctx.prior.prior.ref.settings()
    data = ctx.load_features(source)
    rows = []
    for s, d in data.items():
        with np.load(d['file'], allow_pickle=False) as z:
            d['proposal_inputs'] = nested.data_slice(d, dict(recording=z['recording']))
    for name, c in parent['cells'].items():
        exclude = sorted(c['packets'])
        tag = old.double_name(exclude, c['seed'])
        ref = forecasts['double_excluded'][tag]
        net = base.load_net(old.LOCAL/f'{tag}.pt', ref, old.motion.motion.network(17), exclude)
        for s, p in c['packets'].items():
            d = data[s]['proposal_inputs']
            m = action.study.menu(d, net)
            with screen.packet_file(ctx.study.LOCAL, p) as z:
                np.testing.assert_array_equal(m['cv'], z['cv'])
                np.testing.assert_array_equal(list(m['slots'].values()), z['bounds'])
                a = z['additive_rows']
                row = compare(m['paths'][a], m['peak'][a] > 0., z['additive_paths'])
            rows.append(dict(site=s, seed=c['seed'], **row))
    if len(rows) != 126 or sum(r['views'] for r in rows) != 8874:
        raise ValueError('Changed frozen evaluation roster')

    def summary(r):
        return {**{k: sum(x[k] for x in r) for k in ('actors', 'views', 'ambiguous', 'actual_same', 'uniform_same')},
                **{k: np.asarray([x[k] for x in r]).sum(axis=0).tolist() for k in ('actual', 'uniform')}}

    result = dict(result_source='fresh_input_only_audit', families=['ols4', 'linear_menu_index_2', 'learned_forecast'],
        summary=summary(rows), by_site={s: summary([r for r in rows if r['site'] == s]) for s in base.SITES},
        packets_sha256=sha(ctx.study.OUT/'freeze.json'), code_sha256=sha(Path(__file__)),
        zero_outcomes_materialized=True, external_outcomes_read=False, used_for_tuning=False)
    write(OUT/'anchor_shift.json', result, immutable=True)
    print(result['summary'], flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    main()
