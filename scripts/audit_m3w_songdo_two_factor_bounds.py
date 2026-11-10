"""Replay the Songdo four-cell contrast using a shared missing target."""

from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_songdo_fixed_recipient_paths as original
from src.evaluation import m3w_two_factor_bounds as bounds

np, sha, read = original.np, original.sha, original.read
OUT = original.OUT.parent/'songdo_two_factor_bounds_v1'
LOCAL = original.LOCAL.parent/OUT.name
PARTS = ('total', 'path', 'recipient')
FIELDS = ('observed_pp', 'lower_pp', 'upper_pp', 'missing_radius_pp', 'separate_pair_radius_pp')


def flight(data, name, proposal, previous):
    item, cache, past, bank = data.observed(name)
    with (np.load(past, allow_pickle=False) as p, np.load(bank, allow_pickle=False) as b,
          np.load(proposal, allow_pickle=False) as packet):
        cv, off = b['cv'], b['offsets']
        np.testing.assert_array_equal(p['offsets'], off)
        y, valid = data.targets(name, cache, p, item)
        w = np.repeat(1./np.diff(off), np.diff(off))
        mass = float(w @ np.where(valid, np.linalg.norm(cv-y, axis=-1), 0.).sum(1))
        result = {}
        for seed in original.donor.base.SEEDS:
            k = f'seed{seed}_'
            lo, hi = packet[k+'bounds']
            pred = {}
            for c in original.CELLS:
                p = cv[lo:hi].copy()
                rows = packet[k+c+'_rows']-lo
                if np.any((rows < 0) | (rows >= len(p))) or len(set(rows)) != 2:
                    raise ValueError('Frozen edit is outside its registered query')
                p[rows] = packet[k+c+'_paths']
                pred[c] = p
            x = bounds.contrast(*(pred[c] for c in original.CELLS), y[lo:hi], valid[lo:hi], w[lo:hi], mass)
            prior = previous[str(seed)]['contrasts']
            if x is None:
                raise ValueError('Previously estimated Songdo contrast lost all reference mass')
            for field, old_field in zip(FIELDS[:3], ('known_pp', 'lower_pp', 'upper_pp'), strict=True):
                np.testing.assert_allclose(x['total'][field], prior['diagonal'][old_field], rtol=0., atol=1e-12)
            np.testing.assert_allclose(x['path']['observed_pp'], .5*(prior['path_on_independent']['known_pp']+
                                       prior['path_on_joint']['known_pp']), rtol=0., atol=1e-12)
            np.testing.assert_allclose(x['recipient']['observed_pp'], .5*(prior['recipient_with_independent']['known_pp']+
                                       prior['recipient_with_joint']['known_pp']), rtol=0., atol=1e-12)
            result[str(seed)] = x
    return result


def summarize(rows, groups):
    sources = defaultdict(list)
    for n, c in rows.items():
        sources[groups[n]].append(c)
    if len(sources) != 14 or len(rows) != 160 or any(len(c) != 3 for c in rows.values()):
        raise ValueError('Incomplete frozen Songdo cohort')
    values = [sources[k] for k in sorted(sources)]
    draws = np.random.default_rng(1439).integers(len(values), size=(3000, len(values)))
    effects = {}
    for a in PARTS:
        effects[a] = {}
        for k in FIELDS:
            v = np.array([np.mean([np.mean([s[a][k] for s in r.values()]) for r in g]) for g in values])
            effects[a][k] = dict(point=float(v.mean()), ci95=np.quantile(v[draws].mean(1), [.025, .975]).tolist(),
                                 source_means=v.tolist())
    return effects


def run():
    data, groups, _, base_binding, _ = original.base.check_scope()
    p, r = (original.OUT/f'{k}.json' for k in ('freeze', 'readout'))
    frozen, primary = read(p), read(r)
    if (frozen['binding'] != original.phase_binding(base_binding) or set(frozen['packets']) != set(groups)
            or primary['proposal_freeze_sha256'] != sha(p) or not primary['original_diagonal_reproduced']
            or primary['new_cohort_opened'] or primary['calibration_outcomes_opened']):
        raise ValueError('Require the verified previously scored crossed proposals')
    binding = dict(previous=sha(r), freeze=sha(p), evaluator=sha(Path(bounds.__file__)),
                   driver=sha(Path(__file__)), protocol=sha(OUT/'protocol.md'))
    rows = {}
    LOCAL.mkdir(parents=True, exist_ok=True)
    for n in sorted(groups):
        stem = Path(n).stem
        a = original.base.primary.checked(original.LOCAL/f'{stem}.npz', frozen['packets'][n])
        q = original.LOCAL/f'{stem}_score.json'
        b = read(q)
        if b['identity']['proposal'] != sha(a) or b['identity']['freeze'] != sha(p):
            raise ValueError('Prior fixed-recipient score does not bind the freeze')
        out = LOCAL/f'{stem}.json'
        identity = dict(binding=binding, flight=n, old=sha(q))
        if not out.exists():
            original.storage.write_json(out, dict(identity=identity, effects=flight(data, n, a, b['results'])))
        v = read(out)
        if v['identity'] != identity:
            raise ValueError('Changed per-flight scalar receipt')
        rows[n] = v['effects']
        if len(rows) % 20 == 0:
            print(f'Attributed {len(rows)}/160 Songdo flights', flush=True)
    effects = summarize(rows, groups)
    for k, p in zip(FIELDS[:3], ('known_pp', 'lower_pp', 'upper_pp'), strict=True):
        np.testing.assert_allclose(effects['total'][k]['source_means'], primary['effects']['diagonal'][p]['source_means'],
                                   rtol=0., atol=1e-12)
    original.base.published.secondary.write(OUT/'readout.json', dict(binding=binding, effects=effects,
        flights=160, seeds=3, source_components=14, bootstrap_draws=3000, diagonal_reproduced=True,
        result_source='fresh_retrospective_attribution', previously_opened_outcomes_reread=True,
        new_cohort_opened=False, calibration_outcomes_read=False, original_2pct=True, deployment_changed=False))


if __name__ == '__main__':
    run()
