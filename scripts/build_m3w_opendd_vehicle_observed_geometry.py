"""Recover query-relative OpenDD origins at the exact frozen past slots."""

from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import fcntl
import json
import re
import shutil
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_vehicle_residual as source

np, prior = source.np, source.prior
OUT = source.OUT.parent / 'opendd_vehicle_observed_geometry_v1'
LOCAL = source.LOCAL.parent / OUT.name


def past_coordinates(db, table, ids):
    rows = {}
    for lo in range(0, len(ids), 512):
        part = ids[lo:lo+512]
        query = f'SELECT rowid, UTM_X, UTM_Y FROM "{table}" WHERE rowid IN ({",".join("?" for _ in part)})'
        rows.update((i, (x, y)) for i, x, y in db.execute(query, part.tolist()))
    if set(rows) != set(ids):
        raise ValueError('Observed coordinate row missing')
    return np.array([rows[int(i)] for i in ids], dtype=np.float64)


def observed(db, table):
    if not re.fullmatch(r'rdb[1-7]_\d+', table):
        raise ValueError('Unexpected recording table')
    start, end = source.grid_indices(db.execute(f'SELECT MIN(TIMESTAMP),MAX(TIMESTAMP) FROM "{table}"').fetchone())[0]
    sql = (f'SELECT rowid, OBJID, TIMESTAMP FROM "{table}" WHERE CLASS IN (?,?,?,?,?) '
           'ORDER BY OBJID,TIMESTAMP')
    raw = np.asarray(db.execute(sql, source.CLASSES).fetchall(), dtype=np.float64).reshape(-1, 3)
    if not len(raw):
        return None
    frame = source.grid_indices(raw[:, 2])[0]-start
    agent, begin, _, take = source.indices.exact_indices(raw[:, 1].astype(np.int64), frame,
                                                        raw[:, 0].astype(np.int64), int(end-start))
    if not len(agent):
        return None
    u, ix = np.unique(take[:, :8], return_inverse=True)
    h = past_coordinates(db, table, u)[ix].reshape(len(agent), 8, 2)
    if not np.isfinite(h).all():
        raise ValueError('Nonfinite observed coordinate')
    _, count = np.unique(begin, return_counts=True)
    first = np.r_[0, count.cumsum()[:-1]]
    current = h[:, -1]
    return agent, count, (h-current[:, None]).astype(np.float32), current-np.repeat(current[first], count, axis=0)


def build():
    key, manifest = source.register()
    d = source.load(key)
    info = json.loads((source.OUT / 'cohorts.json').read_text())
    refs = {'script': prior.sha(Path(__file__)), 'cohort_manifest': prior.sha(source.OUT / 'cohorts.json')}
    output = {}
    for src in manifest:
        site = src['site']
        receipt = LOCAL / f'{site}.json'
        path = LOCAL / f'{site}.npz'
        if receipt.exists():
            r = json.loads(receipt.read_text())
            if r['inputs'] != refs or prior.sha(path) != r['sha256']:
                raise ValueError('Observed geometry cache changed')
            output[site] = r
            continue
        archive = prior.ARCHIVES / src['asset']['archive']
        if prior.sha(archive) != src['asset']['sha256']:
            raise ValueError('Source archive changed')
        cohort = d[site]
        site_origins, site_counts = [], []
        with ZipFile(archive) as z, TemporaryDirectory(dir=LOCAL) as tmp:
            member, = [n for n in z.namelist() if Path(n).name == f'trajectories_{site}_v3.sqlite']
            db = Path(tmp) / 'tracks.sqlite'
            with z.open(member) as s, db.open('wb') as o:
                shutil.copyfileobj(s, o, 1024*1024)
            with closing(sqlite3.connect(f'{db.as_uri()}?mode=ro&immutable=1', uri=True)) as conn:
                conn.execute('PRAGMA trusted_schema=OFF')
                conn.execute('PRAGMA query_only=ON')
                for table in dict.fromkeys(cohort['recording']):
                    at = cohort['recording'] == table
                    agent, counts, h, offsets = observed(conn, table)
                    np.testing.assert_array_equal(agent, cohort['agent'][at])
                    np.testing.assert_array_equal(h, cohort['history'][at])
                    site_origins.append(offsets)
                    site_counts.append(counts)
        counts = np.concatenate(site_counts)
        np.testing.assert_array_equal(counts, cohort['query_counts'])
        origins = np.concatenate(site_origins)
        np.testing.assert_array_equal(origins[np.r_[0, counts.cumsum()[:-1]]], 0.)
        sha = source.indices.save_npz(path, dict(observed_current_offsets=origins))
        r = dict(inputs=refs, sha256=sha, cohort_sha256=info['sites'][site]['sha256'],
                 archive_sha256=src['asset']['sha256'], rows=len(origins), queries=len(counts),
                 unique_recordings=len(site_counts), maximum_offset_native=float(np.linalg.norm(origins, axis=1).max()))
        prior.write_json(receipt, r, immutable=True)
        output[site] = r
        print(site, r['rows'], r['queries'], flush=True)
    prior.write_json(OUT / 'manifest.json', dict(result_source='fresh_run', sites=output,
        past_coordinates_only=True, no_labels_exported=True, same_recordings_and_slots=True,
        old_cohort_changed=False, new_model_scored=False, no_external_outcomes_read=True), immutable=True)


if __name__ == '__main__':
    LOCAL.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (LOCAL / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build()
