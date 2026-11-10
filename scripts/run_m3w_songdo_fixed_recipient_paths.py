"""Cross original recipient sets with fixed independent and joint path rules."""

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_songdo_recipient_attribution as base
from scripts import build_m3w_songdo_matched_proposals as donor
from src.evaluation import m3w_fixed_recipient_path_choice as chooser

np, torch, storage = donor.np, donor.torch, donor.pack
ROOT = base.published.ROOT
OUT = base.OUT.parent/'songdo_fixed_recipient_paths_v1'
LOCAL = base.LOCAL.parent/OUT.name
RULES = ('independent', 'joint')
CELLS = ('II', 'IJ', 'JI', 'JJ')
PAIRS = dict(path_on_independent=('IJ', 'II'), path_on_joint=('JJ', 'JI'),
             recipient_with_independent=('JI', 'II'), recipient_with_joint=('JJ', 'IJ'), diagonal=('JJ', 'II'))
METRICS = ('known_pp', 'lower_pp', 'upper_pp', 'relative_available_native', 'relative_lower_native')
read, sha = base.read, storage.sha


def phase_binding(original):
    return dict(original=original, driver=sha(Path(__file__)), selector=sha(Path(chooser.__file__)),
                protocol=sha(OUT/'protocol.md'), proposal_builder=sha(Path(donor.__file__)))


@torch.inference_mode()
def crossed(history, forecasts, packet, seed, ensemble):
    k = f'seed{seed}_'
    lo, hi = packet[k+'bounds']
    s = slice(lo, hi)
    cv, ols, h = forecasts['cv'][s], forecasts['ols4'][s], history[s]
    raw = np.stack([forecasts[f][s] for f in ('ols4', 'median4', f'self_{seed}')], axis=1)
    paths = donor.old.matched.same_radius(cv, raw, np.array([0, hi-lo]))
    peak = np.linalg.norm(paths-cv[:, None], axis=-1).max(axis=-1)
    x, units, _, _ = donor.old.bridge.features(h, cv, ols, paths)
    model = dict(cv=cv, paths=paths, peak=peak, units=units, speed=np.diff(h[:, -4:], axis=1).mean(axis=1))
    row, family, i, j = donor.learn.choices(peak, 0, len(h))
    recipients = {a: packet[k+a+'_rows']-lo for a in RULES}
    keep = np.zeros(len(i), dtype=bool)
    for r in recipients.values():
        keep |= np.isin(row[i], r) & np.isin(row[j], r)
    i, j = i[keep], j[keep]
    rel = donor.learn.relative(model, row[i], row[j], family[i], family[j])
    additive, joint = donor.logits(ensemble, torch.from_numpy(x[row, family]), i, j, torch.from_numpy(rel))
    a = donor.learn.model.merit(additive)
    scores = dict(independent=a[i]+a[j], joint=donor.learn.model.merit(joint))
    radius = float(packet[k+'radius'].item())
    result = dict(bounds=np.array([lo, hi]), radius=radius)
    for group, r in recipients.items():
        for rule, (v, score) in chooser.select(row, i, j, scores, r).items():
            tag = group[0].upper()+rule[0].upper()
            p, f = row[v], family[v]
            np.testing.assert_allclose(peak[p, f], radius, atol=1e-9, rtol=1e-9)
            if group == rule:
                np.testing.assert_array_equal(p+lo, packet[k+rule+'_rows'])
                np.testing.assert_array_equal(f, packet[k+rule+'_families'])
                np.testing.assert_allclose(paths[p, f], packet[k+rule+'_paths'], atol=1e-12, rtol=0.)
                np.testing.assert_allclose(score, packet[k+rule+'_rank_score'].item(), atol=2e-6, rtol=0.)
            result.update({f'{tag}_rows': p+lo, f'{tag}_families': f, f'{tag}_paths': paths[p, f]})
    return result


def heartbeat(phase, n, total):
    p = LOCAL/'heartbeat.pending'
    p.write_text(json.dumps(dict(pid=os.getpid(), phase=phase, completed=n, total=total))+'\n')
    p.replace(LOCAL/'heartbeat.json')
    if n == 1 or n % 20 == 0:
        print(f'{phase}: {n}/{total}', flush=True)


def freeze(data, names, binding):
    _, _, heads, frozen = donor.setup()
    if frozen != read(donor.OUT/'receipt.json')['binding']:
        raise ValueError('Heads are not the original frozen donor ensembles')
    files = {}
    for name in sorted(names):
        _, _, hp, fp = data.observed(name)
        stem = Path(name).stem
        original = base.primary.checked(donor.LOCAL/f'{stem}.npz', data.proposals[name]['packet_sha256'])
        dest, rec = (LOCAL/f'{stem}.{k}' for k in ('npz', 'json'))
        ident = dict(binding=binding, proposal_sha256=sha(original), past_sha256=sha(hp), forecasts_sha256=sha(fp))
        if not rec.exists():
            out = {}
            with (np.load(hp, allow_pickle=False) as p, np.load(fp, allow_pickle=False) as forecasts,
                  np.load(original, allow_pickle=False) as z):
                np.testing.assert_array_equal(p['offsets'], forecasts['offsets'])
                for seed in donor.base.SEEDS:
                    a = crossed(p['history'], forecasts, z, seed, heads[seed])
                    out.update({f'seed{seed}_{k}': v for k, v in a.items()})
            storage.write_json(rec, dict(identity=ident, packet=storage.save_npz(dest, out)))
        item = read(rec)
        if item['identity'] != ident or item['packet'] != sha(dest):
            raise ValueError('Frozen crossed proposal changed')
        files[name] = item['packet']
        heartbeat('freeze', len(files), len(names))
    base.published.secondary.write(OUT/'freeze.json', dict(binding=binding, packets=files, original_diagonals_replayed=True,
        future_coordinates_read=False, before_attribution_scoring=True, prospective_external_test=False))


def require_published(path):
    commit, ref = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=ROOT, timeout=25).split()
    if ref != b'refs/heads/main' or subprocess.check_output(
            ['git', 'show', f'{commit.decode()}:{path.relative_to(ROOT)}'], cwd=ROOT, timeout=10) != path.read_bytes():
        raise ValueError('Publish the frozen counterfactual choices before opening previously scored outcomes')


def readout(data, name, path, original):
    item, cache, prefix, bank = data.observed(name)
    readout = base.primary.protocol.readout
    with (np.load(prefix, allow_pickle=False) as p, np.load(bank, allow_pickle=False) as b,
          np.load(path, allow_pickle=False) as packet):
        cv, ols, off = b['cv'], b['ols4'], b['offsets']
        y, v = data.targets(name, cache, p, item)
        weight = np.repeat(1./np.diff(off), np.diff(off))
        mass = float(weight @ np.where(v, np.linalg.norm(cv-y, axis=-1), 0.).sum(1))
        result = {}
        for seed in donor.base.SEEDS:
            key = f'seed{seed}_'
            lo, hi = packet[key+'bounds']
            radius = float(packet[key+'radius'].item())
            edits = {a: packet[key+a+'_rows'] for a in CELLS}
            for a, other in (('II', 'IJ'), ('JI', 'JJ')):
                if set(edits[a]) != set(edits[other]):
                    raise ValueError('Recipient sets changed within a matched comparison')
            policy = {}
            for a in CELLS:
                rows, paths = edits[a], packet[key+a+'_paths']
                np.testing.assert_allclose(np.linalg.norm(paths-cv[rows], axis=-1).max(1), radius, atol=1e-9, rtol=0.)
                policy[a] = cv.copy()
                policy[a][rows] = paths
            paired = {}
            for a, (left, right) in PAIRS.items():
                pp, rr = policy[left], policy[right]
                r = readout.compare(pp, rr, y, v, weight, mass, (lo, hi))
                reverse = readout.compare(rr, pp, y, v, weight, mass, (lo, hi))
                paired[a] = dict(r, upper_pp=-reverse['lower_pp'], changed=bool(np.any(pp != rr)))
            for k, value in original['scores'][str(seed)]['matched']['matched_contrasts']['independent'].items():
                np.testing.assert_allclose(value, paired['diagonal'][k], atol=1e-12, rtol=0.)
            risk = {}
            for a, pred in policy.items():
                q = readout.policy(pred, cv, cv, y, v, np.repeat('flight', len(cv)),
                                   easy_cut=original['easy_cut'], hard_cut=original['hard_cut'])['recordings']['flight']
                lower = [readout.compare(pred, f, y, v, weight, mass, None)['lower_pp'] for f in (cv, ols)]
                risk[a] = dict(supported=q['supported'], dual=bool(q['supported'] and min(lower) > 0.))
            result[str(seed)] = dict(contrasts=paired, risk=risk)
        return result


def report(rows, names, original, binding, frozen):
    groups = {g: sorted(n for n, v in names.items() if v == g) for g in sorted(set(names.values()))}
    draws = np.random.default_rng(1439).integers(len(groups), size=(3000, len(groups)))
    views = [v for r in rows.values() for v in r.values()]
    effects, support = {}, {}
    for name in PAIRS:
        support[name] = sum(r['contrasts'][name]['changed'] for r in views)
        effects[name] = {}
        for k in METRICS:
            if any(x['contrasts'][name][k] is None for x in views):
                effects[name][k] = None
                continue
            v = np.array([np.mean([np.mean([s['contrasts'][name][k] for s in rows[n].values()]) for n in files])
                          for files in groups.values()])
            effects[name][k] = dict(point=float(v.mean()), ci95=np.quantile(v[draws].mean(1), [.025, .975]).tolist(),
                                     source_means=v.tolist())
    for k, v in original['matched_joint_minus']['independent'].items():
        r = effects['diagonal'][k]
        np.testing.assert_allclose(r['source_means'], v['component_points'], atol=1e-12, rtol=0.)
        np.testing.assert_allclose([r['point'], *r['ci95']], [v['point'], *v['ci95']], atol=1e-12, rtol=0.)
    risk = {a: {k: sum(v['risk'][a][k] for v in views) for k in ('supported', 'dual')} for a in CELLS}
    for a, old in (('II', 'independent'), ('JJ', 'joint')):
        q = original['matched'][old]['risk']
        if (risk[a]['supported'], risk[a]['dual']) != (q['original_risk_pass'], q['dual_pass']):
            raise ValueError('Original diagonal risk counts changed')
    base.published.secondary.write(OUT/'readout.json', dict(result_source='fresh_retrospective_fixed_recipient_experiment',
        binding=binding, proposal_freeze_sha256=frozen, flights=len(rows), flight_seed_views=len(views),
        sources=len(groups), bootstrap_draws=len(draws), effects=effects, differing_paths=support, risk=risk,
        original_diagonal_reproduced=True, original_2pct=True, previously_scored_outcomes_reread=True,
        calibration_outcomes_opened=False, new_cohort_opened=False, training=False, deployment_changed=False))
    print(dict(views=len(views), differing_paths=support, risk=risk), flush=True)


def evaluate(data, names, hashes, binding, original):
    file = OUT/'freeze.json'
    f, freeze_sha = read(file), sha(file)
    if f['binding'] != binding or set(f['packets']) != set(names) or f['future_coordinates_read']:
        raise ValueError('Need the verified outcome-blind crossed choices')
    require_published(file)
    rows = {}
    for n in sorted(names):
        stem = Path(n).stem
        p = base.primary.checked(LOCAL/f'{stem}.npz', f['packets'][n])
        old = base.verified_flight(base.published.RUN/'confirmation'/f'{stem}.json', hashes[n])
        r = LOCAL/f'{stem}_score.json'
        ident = dict(proposal=f['packets'][n], freeze=freeze_sha, original=hashes[n])
        if not r.exists():
            storage.write_json(r, dict(identity=ident, results=readout(data, n, p, old)))
        stored = read(r)
        if stored['identity'] != ident:
            raise ValueError('Retrospective score receipt changed')
        rows[n] = stored['results']
        heartbeat('evaluate', len(rows), len(names))
    report(rows, names, original, binding, freeze_sha)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('freeze', 'score'))
    arg = p.parse_args()
    if sys.platform == 'darwin' and platform.machine() != 'arm64':
        raise SystemExit('Use native arm64 .venv-pytorch')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data, names, old, original_binding, original = base.check_scope()
        binding = phase_binding(original_binding)
        if arg.phase == 'freeze':
            freeze(data, names, binding)
        else:
            evaluate(data, names, old, binding, original)
