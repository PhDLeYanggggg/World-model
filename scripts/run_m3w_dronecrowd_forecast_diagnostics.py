"""Replay secondary calibration ADE/FDE and matched-control pair geometry."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_dronecrowd_calibration as cal
from scripts import report_m3w_dronecrowd_calibration as verified
from src.evaluation import m3w_calibration_forecast_diagnostics as metric

run, np = cal.run, cal.np
PUBLIC = cal.PUBLIC.parent/'dronecrowd_forecast_diagnostics_v1'
PRIVATE = cal.PRIVATE.parent/PUBLIC.name
REG = PUBLIC/'registration.json'
FILES = (Path(__file__).resolve(), Path(metric.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_calibration_forecast_diagnostics.py')


def register():
    cal.register()
    inputs = (*FILES, cal.REG, cal.roles.DEST, cal.CACHE, Path(verified.__file__), Path(cal.__file__),
              Path(metric.local_edges.__code__.co_filename))
    run.once(REG,run.stable(dict(bindings=[run.artifact(f) for f in inputs], scope='calibration_secondary_only',
        requires_full_official_replay=True, no_new_threshold_selection=True, seeds=cal.SEEDS, draws=3000,
        matched_controls=metric.MATCHED, official_rule_unchanged=True, confirmation_access=False)))


def unpack(f, seed):
    base, offsets = f['cv'], metric.validate_offsets(f['offsets'],len(f['cv']))
    paths = {cal.BASE:base, cal.OLS:f['ols']}
    arms, counts = cal.model.ARMS, {}
    slots, paths_at = f[f'index_{seed}'], f[f'paths_{seed}']
    for a,i,p in zip(arms,slots,paths_at,strict=True):
        use = i >= 0
        lo, hi = offsets[:-1,None], offsets[1:,None]
        if np.any(use & ((i < lo) | (i >= hi))) or len(np.unique(i[use])) != use.sum():
            raise ValueError('Frozen action slots do not belong to unique query rows')
        candidate = base.copy()
        candidate[i[use]] = p[use]
        counts[a] = (use & np.any(candidate[np.maximum(i,0)] != base[np.maximum(i,0)],axis=(-2,-1))).sum(1)
        paths[a] = candidate
    for a in metric.MATCHED[1:]:
        np.testing.assert_array_equal(counts[metric.MATCHED[0]],counts[a])
    return paths, {a:int(n.sum()) for a,n in counts.items()}


def score(official, binds, beat):
    manifest, cut = run.read(cal.roles.DEST), run.read(cal.REG)['cut']
    freeze = run.read(cal.PUBLIC/'forecast_freeze.json')
    inventory = run.read(cal.CACHE)
    roster = {r:g for r,g in run.read(cal.roles.GROUPS)['recording_group'].items()
              if manifest['recording_role_index'][r] == 'calibration'}
    if len(roster) != 40 or len(set(roster.values())) != 15 or set(roster) != {c['sequence'] for c in freeze['clips']}:
        raise ValueError('Secondary score must use the complete calibration roster only')
    meta = {r['sequence_id']:r for r in inventory['recordings'] if r['sequence_id'] in roster}
    seen, comparisons, switches, records = [], [], dict.fromkeys(cal.model.ARMS,0), []
    for c in freeze['clips']:
        rec = c['sequence']
        source = cal.reader(manifest,rec,meta,inventory['provenance'])
        with np.load(run.checked(c['artifact']),allow_pickle=False) as a:
            f = {k:a[k] for k in a.files}
        np.testing.assert_array_equal(f['frames'],source.query_frames)
        y, valid = cal.label_arrays(source,f)
        edges = metric.graph_queries(f['origin'],f['offsets'])
        coverage = counts = None
        for s in cal.SEEDS:
            paths, used = unpack(f,s)
            table, support = metric.individual(paths,y,valid,f['offsets'],easy_cut=cut['easy'],hard_cut=cut['hard'])
            pair, size = metric.relative({k:paths[k] for k in metric.MATCHED},y,valid,edges)
            seen.append((rec,s,table)); comparisons.append((rec,s,pair))
            for a,n in used.items():
                switches[a] += n
            if coverage is not None and (coverage != size or counts != support):
                raise ValueError('Target support differs by seed')
            coverage, counts = size,support
        records.append(dict(recording=rec,source=roster[rec],rows=len(valid),queries=len(edges),support=counts,
                            observed_pair_steps=coverage[0],requested_pair_steps=coverage[1]))
        beat(recording=rec,rows=len(valid),completed=len(records))
    if any(n != official['summary']['risk'][a]['switches'] for a,n in switches.items()):
        raise ValueError('Secondary score changed the official intervention counts')
    absolute, gain = metric.aggregate(seen,roster,cal.SEEDS)
    pair, _ = metric.aggregate(comparisons,roster,cal.SEEDS)
    key = lambda x: '__'.join(x)
    return dict(absolute_pixel={key(k):v for k,v in absolute.items()}, gain_vs_cv_pixel={key(k):v for k,v in gain.items()},
        joint_gain_vs_matched_controls_pixel={key(k):v for k,v in pair.items()},
        recordings=records, switches=switches, source_groups=15, views=len(seen), seeds=cal.SEEDS,
        bootstrap_draws=3000, primary_winner_unchanged=official['selected_arm'], official_bindings=binds,
        calibration_only=True, independent_confirmation=False, confirmation_outcome_access=False,
        official_rule_changed=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)


def render(doc):
    def value(r):
        if r['point'] is None:
            return 'NA'
        a,b = r['interval95']
        return f'{r["point"]:+.3f} [{a:+.3f}, {b:+.3f}]'

    arms = (cal.BASE,cal.OLS,*cal.model.ARMS)
    lines = ['# Secondary DroneCrowd calibration diagnostics','','The original decision is unchanged. This is not independent confirmation.',
             '', '| Arm | Available ADE (px) | Available step-12 FDE (px) | Easy ADE (px) | Hard ADE (px) |',
             '| --- | ---: | ---: | ---: | ---: |']
    for a in arms:
        v = [value(doc['absolute_pixel'][f'{a}__{s}__{m}']) for s,m in (
            ('all','ade'),('all','fde'),('easy','ade'),('hard','ade'))]
        lines.append('| '+' | '.join((a,*v))+' |')
    lines += ['', '| Joint vs same-count control | Observed pair gain (px) | Full-grid lower (px) | Full-grid upper (px) |',
              '| --- | ---: | ---: | ---: |']
    for a in metric.MATCHED[:-1]:
        v = [value(doc['joint_gain_vs_matched_controls_pixel'][f'{a}__{k}']) for k in ('observed','grid_lower','grid_upper')]
        lines.append('| '+' | '.join((a,*v))+' |')
    lines += ['', 'Positive pair gain favors joint. Missing steps are bounded by forecast disagreement, not treated as zero error.',
              'Intervals are paired source bootstraps over 15 groups after seed averaging (3,000 draws).',
              'NA denotes missing source-seed support. Units are image pixels and annotation steps.', '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('register','score','replay'))
    a = p.parse_args()
    register()
    if a.phase == 'register':
        return
    for f in (*FILES,REG):
        if subprocess.check_output(['git','show',f'HEAD:{f.relative_to(ROOT)}'],cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit the secondary analysis before scoring')
    official, bindings = verified.verify_artifacts(ROOT,cal.PUBLIC,cal.roles.DEST)
    verified.summarize(official)
    PRIVATE.mkdir(parents=True,exist_ok=True)
    out = PUBLIC/'analysis.json'
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            event = dict(pid=os.getpid(),phase=a.phase,seconds=time.monotonic()-start,**v)
            cal.run.kin.prior.source.atomic(PRIVATE/'heartbeat.json',event)
            print(event,flush=True)

        if a.phase == 'replay' and run.read(PUBLIC/'score_execution.json')['artifact'] != run.artifact(out):
            raise ValueError('Require verified original secondary score')
        run.once(out,run.stable(score(official,bindings,beat)))
        receipt = PUBLIC/f'{a.phase}_execution.json'
        if not receipt.exists():
            run.once(receipt,dict(pid=os.getpid(),seconds=time.monotonic()-start,artifact=run.artifact(out)))
        elif run.read(receipt)['artifact'] != run.artifact(out):
            raise ValueError('Secondary result changed on replay')
        if a.phase == 'replay':
            verified.write_once(PUBLIC/'report.md',render(run.read(out)))


if __name__ == '__main__':
    main()
