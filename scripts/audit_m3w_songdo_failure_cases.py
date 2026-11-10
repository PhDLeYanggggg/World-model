"""Render three median retrospective cases from the frozen Songdo comparison."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_songdo_fixed_recipient_paths as source
from src.evaluation import m3w_failure_case_panels as panels

np = source.np
OUT = source.OUT.parent/'songdo_failure_cases_v1'
LOCAL = source.LOCAL.parent/OUT.name
CASES = dict(recipients=('II', 'JJ', 'diagonal', 'Different recipients'),
             paths=('II', 'IJ', 'path_on_independent', 'Same recipients, different paths'),
             missing=('II', 'JJ', 'diagonal', 'Incomplete future labels'))
SEED = '17'


def roster(data, groups, hashes, frozen, freeze_sha):
    eligible = {k: [] for k in CASES}
    for member in sorted(groups):
        stem = Path(member).stem
        path = source.base.primary.checked(source.LOCAL/f'{stem}.npz', frozen['packets'][member])
        score = source.read(source.LOCAL/f'{stem}_score.json')
        identity = dict(proposal=frozen['packets'][member], freeze=freeze_sha, original=hashes[member])
        if score['identity'] != identity:
            raise ValueError('Unbound crossed score')
        c = score['results'][SEED]['contrasts']
        diagonal, fixed = c['diagonal'], c['path_on_independent']
        path_component = .5*(fixed['known_pp']+c['path_on_joint']['known_pp'])
        recipient_component = diagonal['known_pp']-path_component
        item = data.source[member]
        pack = source.base.primary
        cache = pack.pack.CACHE if item['cache_role'] == 'previous' else pack.first.PRIVATE
        index = pack.checked(cache/f'{stem}_label_index.npz', item['label_index_sha'])
        with np.load(index, allow_pickle=False) as ix, np.load(path, allow_pickle=False) as z:
            complete = ix['future_mask'].all(1)
            i, j = (z[f'seed{SEED}_{cell}_rows'] for cell in ('II', 'JJ'))
            if (set(i) != set(j) and complete[np.union1d(i, j)].all() and diagonal['known_pp'] < 0
                    and abs(recipient_component) >= 2*abs(path_component)):
                eligible['recipients'].append(dict(member=member, rank=abs(diagonal['known_pp']),
                    path_component_pp=path_component, recipient_component_pp=recipient_component))
            if fixed['changed'] and complete[i].all():
                eligible['paths'].append(dict(member=member, rank=abs(fixed['known_pp'])))
        if diagonal['lower_pp'] < 0 < diagonal['known_pp']:
            eligible['missing'].append(dict(member=member, rank=diagonal['known_pp']-diagonal['lower_pp']))
    return eligible


def replay(data, case, candidate, frozen):
    member = candidate['member']
    stem = Path(member).stem
    item, cache, prefix, bank = data.observed(member)
    packet = source.base.primary.checked(source.LOCAL/f'{stem}.npz', frozen['packets'][member])
    left, right, contrast, _ = CASES[case]
    with (np.load(prefix, allow_pickle=False) as p, np.load(bank, allow_pickle=False) as b,
          np.load(packet, allow_pickle=False) as z):
        cv, off = b['cv'], b['offsets']
        np.testing.assert_array_equal(p['offsets'], off)
        y, v = data.targets(member, cache, p, item)
        weight = np.repeat(1./np.diff(off), np.diff(off))
        mass = float(weight @ np.where(v, np.linalg.norm(cv-y, axis=-1), 0.).sum(1))
        key = f'seed{SEED}_'
        l, r = (z[key+a+'_rows'] for a in (left, right))
        u = np.union1d(l, r)
        lo, hi = z[key+'bounds']
        if not (u.min() >= lo and u.max() < hi and np.all(weight[u] == weight[lo])):
            raise ValueError('Edits must lie inside one common weighted query')
        a, b = cv[u].copy(), cv[u].copy()
        a[np.searchsorted(u, l)] = z[key+left+'_paths']
        b[np.searchsorted(u, r)] = z[key+right+'_paths']
        result = panels.compare(a, b, y[u], v[u], 100.*weight[lo]/mass)
        result.update(left_edits=np.isin(u, l).tolist(), right_edits=np.isin(u, r).tolist(),
                      supported_steps=v[u].tolist())
    archived = source.read(source.LOCAL/f'{stem}_score.json')['results'][SEED]['contrasts'][contrast]
    for key, previous in zip(('observed', 'lower', 'upper'), ('known_pp', 'lower_pp', 'upper_pp')):
        np.testing.assert_allclose(result[key][-1], archived[previous], atol=1e-12, rtol=0.)
    result.update({k: v for k, v in candidate.items() if k.endswith('_component_pp')})
    return result


def draw(results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, grid = plt.subplots(3, 2, figsize=(8.1, 5.35), gridspec_kw=dict(width_ratios=[1, 1.3]))
    fig.subplots_adjust(left=.10, right=.97, top=.88, bottom=.17, wspace=.38, hspace=.87)
    colors = ['#ffffff', '#196d86', '#b65c27', '#e4e7ea', '#fbfbfb']
    for pos, ((name, case), (table, graph)) in enumerate(zip(results.items(), grid)):
        if case is None:
            table.text(.05, .5, 'No eligible case', transform=table.transAxes)
            table.set_axis_off()
            graph.set_axis_off()
            continue
        mask = np.array(case['supported_steps'])
        n = len(mask)
        cols = np.c_[np.where(case['left_edits'], 1, 0), np.where(case['right_edits'], 2, 0),
                     np.where(mask, 4, 3)]
        table.pcolormesh(np.arange(15), np.arange(n+1), cols, cmap=ListedColormap(colors),
                         vmin=0, vmax=4, edgecolors='#bbc1c6', linewidth=.55)
        table.set_ylim(n, 0)
        table.set_xticks(np.arange(14)+.5, ['I', 'J', *map(str, range(1, 13))], fontsize=7)
        table.set_yticks(np.arange(n)+.5, list('ABCD')[:n], fontsize=8)
        table.tick_params(length=0)
        table.set_ylabel('Actor', fontsize=8)
        table.set_title(f'{chr(65+pos)}  {CASES[name][3]}', loc='left', fontsize=10, fontweight='bold')
        for s in table.spines.values():
            s.set_visible(False)
        t = np.arange(13)
        y, low, high = (1000*np.array(case[k]) for k in ('observed', 'lower', 'upper'))
        graph.axhline(0, color='#81878b', lw=.65, ls='--')
        graph.fill_between(t, low, high, color='#cbd9e1')
        graph.plot(t, y, color='#18495b', lw=1.65)
        graph.scatter(12, y[-1], s=14, c='#18495b', zorder=3)
        graph.set(xlim=(0, 12.3), xticks=[0, 3, 6, 9, 12])
        graph.tick_params(labelsize=7, length=3, color='#81878b')
        graph.spines[['top', 'right']].set_visible(False)
        if pos == 0:
            graph.set_title('Cumulative paired gain', loc='left', fontsize=10, fontweight='bold')
        if pos == 2:
            graph.set_xlabel('Forecast step', fontsize=8)
        graph.set_ylabel(r'Gain ($10^{-3}$ pp)', fontsize=8)
    fig.text(.10, .95, 'Three ways to read the joint proposal',
             fontsize=13, fontweight='bold')
    fig.legend(handles=[Patch(facecolor=colors[1], label='I edit'), Patch(facecolor=colors[2], label='J edit'),
                        Patch(facecolor=colors[3], label='Missing target'),
                        Line2D([], [], color='#18495b', label='Observed gain'),
                        Patch(facecolor='#cbd9e1', label='Missing-label bounds')],
               loc='lower center', bbox_to_anchor=(.54, .016), ncol=5, frameon=False, fontsize=8)
    fig.savefig(output, metadata=dict(Creator=None, CreationDate=None, ModDate=None))
    plt.close(fig)


def main():
    data, groups, hashes, original_binding, _ = source.base.check_scope()
    freeze_file, receipt = (source.OUT/f'{k}.json' for k in ('freeze', 'readout'))
    frozen, previous = source.read(freeze_file), source.read(receipt)
    freeze_sha = source.sha(freeze_file)
    if (set(frozen['packets']) != set(groups) or frozen['binding'] != source.phase_binding(original_binding)
            or frozen['future_coordinates_read'] or previous['proposal_freeze_sha256'] != freeze_sha
            or not previous['original_diagonal_reproduced'] or previous['new_cohort_opened']):
        raise ValueError('Expected the frozen, already scored crossed comparison')
    eligible = roster(data, groups, hashes, frozen, freeze_sha)
    selected = {k: panels.median_case(v) for k, v in eligible.items()}
    result = {k: replay(data, k, v, frozen) if v else None for k, v in selected.items()}
    LOCAL.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=LOCAL) as tmp:
        figure = Path(tmp)/'failure_cases.pdf'
        draw(result, figure)
        source.storage.write_json(OUT/'readout.json', dict(result_source='fresh_retrospective_case_readout',
            seed=int(SEED), scored_roster=len(groups), eligible={k: len(v) for k, v in eligible.items()},
            numeric_label_joins=sum(v is not None for v in selected.values()), cases=result,
            identity=dict(freeze_sha256=freeze_sha, crossed_readout_sha256=source.sha(receipt),
                          driver_sha256=source.sha(Path(__file__)), arithmetic_sha256=source.sha(Path(panels.__file__)),
                          protocol_sha256=source.sha(OUT/'protocol.md'), figure_sha256=source.sha(figure)),
            new_cohort_opened=False, training=False, policy_changed=False, original_2pct_unchanged=True))
        figure.replace(LOCAL/figure.name)
    print({k: (len(eligible[k]), None if not v else [v[m][-1] for m in ('observed', 'lower', 'upper')])
           for k, v in result.items()}, flush=True)


if __name__ == '__main__':
    main()
