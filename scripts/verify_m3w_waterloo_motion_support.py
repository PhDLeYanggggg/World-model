"""Independently recount continuous pedestrian tracks and support concentration."""

from contextlib import closing
from itertools import groupby
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_m3w_waterloo_intake import ASSETS, PUBLIC, SOURCES, sha, write
import numpy as np


def audit(saved):
    reports = {}
    for site in SOURCES:
        source = saved["sites"][site]
        tracks_per_stride = {s: [] for s in (1, 12)}
        motion_per_stride = {s: [] for s in (1, 12)}
        nonoverlap_counts = {s: 0 for s in (1, 12)}
        archive = ASSETS/f"{site}_dbfiles.zip"
        if sha(archive) != source["archive_sha256"]:
            raise ValueError("Archive content changed")
        with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory(dir=ASSETS) as temp:
            for info in source["recordings"]:
                p = Path(temp)/"work.db"
                with z.open(info["member"]) as f, p.open("wb") as g:
                    shutil.copyfileobj(f, g, length=1024**2)
                counts = {s:np.zeros(4, dtype=np.int64) for s in (1, 12)}
                by_query = {s:[] for s in (1, 12)}
                with closing(sqlite3.connect(p.as_uri()+"?mode=ro&immutable=1", uri=True)) as db:
                    db.execute("PRAGMA trusted_schema=OFF")
                    table = "TRAJECTORIES" if site=="crosswalk" else f'TRAJECTORIES_{int(info["file_id"]):04d}'
                    rows = db.execute(f'SELECT a.TRACK_ID,a.TIME,a.X,a.Y FROM "{table}" AS a '
                                      'JOIN TRACKS AS b USING(TRACK_ID) WHERE b.TYPE=? ORDER BY a.TRACK_ID,a.TIME',
                                      ("Pedestrian",))
                    for _, rows_for_track in groupby(rows, key=lambda x:x[0]):
                        a = np.asarray([r[1:] for r in rows_for_track])
                        t = np.rint(a[:, 0]/info["lattice_step_source_seconds"]).astype(np.int64)
                        if not (np.diff(t) == 1).all():
                            raise ValueError("Continuous-track recount is not applicable")
                        for s in (1, 12):
                            n = max(0, len(t)-7*s)
                            m = max(0, n-12*s)
                            no = min(s, n)
                            counts[s] += (n, m, n-m-no, no)
                            tracks_per_stride[s].append(m)
                            by_query[s].append(t[7*s:7*s+m])
                            if m:
                                past = a[np.arange(7*s,7*s+m)[:, None] - s*np.arange(8)[None, :], 1:]
                                path = np.linalg.norm(np.diff(past, axis=1), axis=2).sum(1)
                                motion_per_stride[s].append(path)
                for s, count in counts.items():
                    reference = info["availability"][str(s)]["pedestrian"]
                    np.testing.assert_array_equal(count, [reference[key] for key in (
                        "past_eligible", "complete_future12", "partial_future12", "no_future12")])
                    if by_query[s]:
                        frame, agents = np.unique(np.concatenate(by_query[s]), return_counts=True)
                        np.testing.assert_equal((agents>=2).sum(), reference["queries_with_two_complete_agents"])
                        prev_end = -1
                        for q in frame[agents>=2]:
                            if q-7*s > prev_end:
                                nonoverlap_counts[s] += 1
                                prev_end = q+12*s
                p.unlink()
                print(site, info["member"], "recounted", flush=True)
        out = {}
        for s in (1, 12):
            m = np.concatenate(motion_per_stride[s])
            a = np.sort(tracks_per_stride[s])
            if len(m) != a.sum():
                raise ValueError("Movement observations do not match full labels")
            out[str(s)] = dict(complete_windows=len(m), tracks_with_complete=int((a>0).sum()),
                top_10_tracks_share=float(a[-10:].sum()/a.sum()),
                past_path_length_p10_p50_p90_source_coordinate=np.quantile(m,[.1,.5,.9]).tolist(),
                zero_past_path_windows=int((m==0).sum()),
                at_most_point1_past_path_windows=int((m<=.1).sum()),
                nonoverlapping_intervals_with_at_least_two_complete_pedestrians=int(nonoverlap_counts[s]))
        reports[site] = out
    return dict(sites=reports, exact_pedestrian_recount=True, parent_analysis_sha256=sha(PUBLIC/"analysis.json"),
                no_model_evaluation=True, overlapping_windows_not_independent=True)


if __name__ == "__main__":
    write(PUBLIC/"motion_support.json", audit(json.loads((PUBLIC/"analysis.json").read_text())))
