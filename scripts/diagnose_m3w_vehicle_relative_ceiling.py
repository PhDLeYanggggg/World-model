"""Audit exact same-budget interaction headroom on exposed OpenDD recordings."""

from collections import Counter
import fcntl
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_vehicle_pair_learning as study
from scripts import run_m3w_opendd_support_rerank as prior
from src.evaluation.m3w_interaction_frontier import INTERACTION_TOL, pair_reduction
from src.evaluation.m3w_recording_switch_ceiling import moments
from src.evaluation.m3w_vehicle_relative_ceiling import relative_lower

base, pair, old, np, torch = study.base, study.pair, study.old, study.np, study.torch
sha, write = prior.sha, prior.write
OUT = study.OUT.parent / 'opendd_vehicle_relative_ceiling_v1'
LOCAL = study.LOCAL.parent / OUT.name
KEYS = ('views', 'legal_pairs', 'any_dual', 'any_same_family_dual', 'any_dual_no_trajectory_loss',
        'any_relative_win', 'any_same_family_relative_win', 'any_keep_recipient_relative_win',
        'additive_dual', 'joint_dual', 'joint_relative_win', 'additive_dual_and_any_relative_win',
        'unrestricted_only_relative_win')


def settings():
    key, source, _, _, forecasts = study.register()
    p, q = (study.OUT/x for x in ('freeze.json', 'readout.json'))
    frozen, scored = old.read(p), old.read(q)
    if (frozen['registration'] != key or scored['freeze'] != sha(p) or not frozen['held_choices_frozen']
            or scored['external_outcomes_read'] or not scored['original_2pct_unchanged']):
        raise ValueError('Expected the scored, exposed vehicle pair study')
    binding = {str(x.relative_to(base.prior.ROOT)): sha(x) for x in (
        Path(__file__), Path(relative_lower.__code__.co_filename), Path(pair_reduction.__code__.co_filename),
        Path(moments.__code__.co_filename), OUT/'protocol.md', p, q)}
    return source, forecasts, frozen, scored, binding


def audit(packet, data, menu, cut):
    cv, ols = packet['cv'], packet['ols4']
    y, mask, weight = (data[k] for k in ('future', 'valid', 'weight'))
    add = pair.prediction(packet, 'additive')
    joint = pair.prediction(packet, 'joint')
    base_ols = base.parent.paired(ols, cv, y, mask)[1].sum(1)
    base_add = base.parent.paired(add, cv, y, mask)[1].sum(1)
    names, rec = np.unique(data['recording'], return_inverse=True)
    ols_offset = np.bincount(rec, weights=weight*base_ols)
    add_offset = np.bincount(rec, weights=weight*base_add)
    cv0 = moments(cv, cv, y, mask, cut[0])
    signed_cap = .02*np.bincount(rec, weights=cv0['ref_easy'])
    add_eval = pair.evaluate(add, packet, data, cut)
    joint_eval = pair.evaluate(joint, packet, data, cut)
    stats = Counter({k: 0 for k in KEYS})
    stats.update(additive_dual=int(add_eval['dual'].sum()), joint_dual=int(joint_eval['dual'].sum()))
    witness, witness_names = cv.copy(), []
    best_relative = []
    for k, (name, (lo, hi)) in enumerate(zip(packet['records'], packet['bounds'], strict=True)):
        q = slice(lo, hi)
        index = np.searchsorted(names, name)
        if names[index] != name or not np.all(rec[q] == index):
            raise ValueError('The frozen query is not a complete single recording window')
        rows, family, first, second = study.choices(menu['peak'], lo, hi)
        p = menu['paths'][rows, family]
        m = moments(cv[rows], p, y[rows], mask[rows], cut[0])
        risk = np.column_stack([m[key] for key in ('all_excess', 'easy_excess', 'easy_signed')])
        gains = np.column_stack((base.parent.paired(cv[rows], p, y[rows], mask[rows])[1].sum(1),
                                 base.parent.paired(ols[rows], p, y[rows], mask[rows])[1].sum(1)-base_ols[rows],
                                 base.parent.paired(add[rows], p, y[rows], mask[rows])[1].sum(1)-base_add[rows]))
        gains *= weight[rows, None]
        g = gains[first]+gains[second]+[0., ols_offset[index], add_offset[index]]
        supported = m['ref_easy'][first]+m['ref_easy'][second] > 0.
        safe = (risk[first]+risk[second] <= [0., 0., signed_cap[index]]).all(1) & supported
        dual = safe & (g[:, :2] > 0.).all(1)
        no_loss = dual & (g[:, 2] >= -1e-10)
        same = family[first] == family[second]
        stats.update(views=1, legal_pairs=len(first), any_dual=int(dual.any()),
                     any_same_family_dual=int((dual & same).any()), any_dual_no_trajectory_loss=int(no_loss.any()))
        joint_delta = base.parent.paired(add[q], joint[q], y[q], mask[q])[1].sum(1) @ weight[q]
        if (joint_eval['dual'][k] and joint_delta >= -1e-10
                and pair_reduction(joint[None, q], add[q], y[q], mask[q])[0, 1] > INTERACTION_TOL):
            stats['joint_relative_win'] += 1
        idx = np.flatnonzero(no_loss)
        if not len(idx):
            continue
        rel = relative_lower(cv[q], add[q], y[q], mask[q], rows-lo, p, first[idx], second[idx])
        wins = rel > INTERACTION_TOL
        if not wins.any():
            continue
        stats['any_relative_win'] += 1
        stats['additive_dual_and_any_relative_win'] += int(add_eval['dual'][k])
        sf = same[idx][wins].any()
        stats['any_same_family_relative_win'] += int(sf)
        stats['unrestricted_only_relative_win'] += int(not sf)
        kept = np.isin(rows[first[idx]], packet['additive_rows'][k]) | np.isin(
            rows[second[idx]], packet['additive_rows'][k])
        stats['any_keep_recipient_relative_win'] += int((kept & wins).any())
        at = int(rel.argmax())
        ij = np.array([first[idx[at]], second[idx[at]]])
        witness[rows[ij]] = p[ij]
        exact = pair_reduction(witness[None, q], add[q], y[q], mask[q])[0, 1]
        np.testing.assert_allclose(rel[at], exact, rtol=0., atol=1e-9)
        if exact <= INTERACTION_TOL:
            raise ValueError('Chosen relative-motion witness is not a strict improvement')
        witness_names.append(name)
        best_relative.append(float(exact))
    if witness_names:
        v = pair.evaluate(witness, dict(cv=cv, ols4=ols, records=np.asarray(witness_names)), data, cut)
        if not v['dual'].all():
            raise ValueError('Oracle witness failed the original risk or reference checks')
        _, lower = base.parent.paired(add, witness, y, mask)
        s = np.bincount(rec, weights=weight*lower.sum(1))
        if np.any(s[np.searchsorted(names, witness_names)] < -1e-9):
            raise ValueError('Oracle witness degraded conservative trajectory gain against additive')
    return dict(counts=dict(stats), mean_best_relative_native=float(np.mean(best_relative)) if best_relative else None)


def run():
    source, forecasts, frozen, scored, binding = settings()
    data, results, records = base.parent.load(source), [], set()
    for name, c in frozen['cells'].items():
        excluded = sorted(c['packets'])
        seed = c['seed']
        if set(c['fit_sites']) != set(base.SITES)-set(excluded):
            raise ValueError('Changed frozen producer exclusions')
        tag = old.double_name(excluded, seed)
        producer = None
        for site, r in c['packets'].items():
            path = LOCAL / f'{name}_{site}.json'
            identity = dict(binding=binding, packet=r['sha256'], site=site, seed=seed)
            with prior.packet(study.LOCAL, r) as z:
                records.update(z['records'].tolist())
                if not path.exists():
                    if producer is None:
                        producer = base.load_net(old.LOCAL / f'{tag}.pt', forecasts['double_excluded'][tag],
                                                 old.motion.motion.network(17), excluded)
                    menu = study.menu(data[site], producer)
                    for k in ('cv', 'ols4'):
                        np.testing.assert_array_equal(menu[k], z[k])
                    np.testing.assert_array_equal(list(menu['slots'].values()), z['bounds'])
                    write(path, dict(identity=identity, **audit(z, data[site], menu, c['cut'])), immutable=True)
            found = old.read(path)
            if found['identity'] != identity:
                raise ValueError('Resumed oracle cell does not bind frozen study')
            results.append(found)
        write(LOCAL/'heartbeat.json', dict(pid=os.getpid(), phase='relative_ceiling', complete=len(results), cell=name))
        print(name, len(results), flush=True)
    totals = {k: sum(r['counts'][k] for r in results) for k in KEYS}
    if (len(results) != 126 or len(records) != 493 or totals['views'] != 8874
            or totals['additive_dual'] != scored['summary']['additive']['matched_dual']
            or totals['joint_dual'] != scored['summary']['joint']['matched_dual']
            or totals['joint_relative_win'] > totals['any_relative_win']):
        raise ValueError('Oracle roster or original control counts did not replay')
    by_site = {s: {k: sum(r['counts'][k] for r in results if r['identity']['site'] == s) for k in KEYS} for s in base.SITES}
    draws = np.random.default_rng(2917).integers(len(base.SITES), size=(3000, len(base.SITES)))
    rates = {}
    for k in ('any_dual', 'any_relative_win', 'any_same_family_relative_win', 'joint_relative_win',
              'unrestricted_only_relative_win'):
        x = np.array([100.*r[k]/r['views'] for r in by_site.values()])
        rates[k] = dict(site_macro_percent=float(x.mean()), ci95=np.quantile(x[draws].mean(1), [.025, .975]).tolist())
    write(OUT/'readout.json', dict(result_source='fresh_run', binding=binding, totals=totals, by_site=by_site,
        rate_intervals=rates, unique_recordings=len(records), directions=len(results),
        label_aware_not_deployable=True, original_2pct_unchanged=True, external_outcomes_read=False,
        thresholds_changed=False, independent_confirmation=False), immutable=True)
    print(totals, flush=True)


if __name__ == '__main__':
    LOCAL.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
