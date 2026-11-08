"""Check the shortest publicly indexed roundabout raw/SQLite candidate pair."""

from collections import Counter
from contextlib import closing
from itertools import groupby
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from scripts.run_m3w_waterloo_raw_prefix import DATA, PUBLIC, asset, immutable
from src.evaluation.m3w_waterloo_intake import sha, support, ticks


def run():
    zip_path = DATA/"dataset.zip"
    video = DATA/"DJI_0932.MP4"
    origin = "https://wiselab.uwaterloo.ca/multi-agent-roundabout-traffic/"
    facts = {
        "annotation_archive": asset(zip_path, origin+"dataset/dataset.zip", 210366619),
        "raw_video": asset(video, origin+"raw_videos/2021.2.5%20recording/"+video.name, 635003549),
    }
    with zipfile.ZipFile(zip_path) as z, tempfile.TemporaryDirectory(dir=DATA) as t:
        match = re.fullmatch(r"DJI_(\d+)\.MP4", video.name)
        if match is None:
            raise ValueError("Unrecognized publisher raw name")
        member = f"{int(match[1])}.db"
        ids = sorted(z.namelist())
        p = Path(t)/"check.db"
        with z.open(member) as f, p.open("wb") as g:
            shutil.copyfileobj(f,g,length=1024**2)
        facts["member_sha256"] = sha(p)
        with closing(sqlite3.connect(p.as_uri()+"?mode=ro&immutable=1",uri=True)) as db:
            db.execute("PRAGMA trusted_schema=OFF")
            db.execute("PRAGMA query_only=ON")
            meta = db.execute("SELECT FILE_ID, TRACK_ID, TYPE FROM TRACKS").fetchall()
            if {int(row[0]) for row in meta} != {932}:
                raise ValueError("SQLite FILE_ID is not the candidate raw-video index")
            category = {tid:ty for _,tid,ty in meta}
            if len(category) != len(meta):
                raise ValueError("Track IDs not unique")
            rows = db.execute("SELECT TRACK_ID,TIME FROM TRAJECTORIES ORDER BY TRACK_ID,TIME")
            tracks = [(category[agent], np.asarray([r[1] for r in block], dtype=float))
                      for agent,block in groupby(rows,key=lambda x:x[0])]
            xy = {n for n, in db.execute("SELECT name FROM pragma_table_info('TRAJECTORIES')")}
    tau,jitter = ticks([r for _,r in tracks])
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError("Raw MP4 not readable")
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or frames <= 0:
            raise ValueError("Missing container clock or frame count")
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0,frames-1))
        last_decoded, _ = cap.read()
    finally:
        cap.release()
    availability = {}
    for cadence in (1,4,12):
        pedestrian = Counter()
        if tau is not None:
            for ty,t in tracks:
                if ty == "Pedestrian":
                    count,_,_ = support(np.rint(t/tau).astype(np.int64),cadence)
                    pedestrian.update(count)
        availability[str(cadence)] = dict(pedestrian) if tau is not None else None
    return dict(assets=facts, database_member=member, archive_databases=len(ids), file_id_verified=932,
        class_track_counts=Counter(ty for ty,_ in tracks), label_timestamp_rows=sum(len(t) for _,t in tracks),
        min_timestamp=float(min(t.min() for _,t in tracks)), max_timestamp=float(max(t.max() for _,t in tracks)),
        source_seconds_per_tick=tau, lattice_residual_seconds=jitter, container_fps=fps,
        container_frames=frames, container_last_frame_decode_ok=bool(last_decoded),
        sample_DJI_0669_in_sqlite_archive="669.db" in ids, pixel_box_columns_present=bool(
            {"IMAGE_X","IMAGE_Y","BOX_WIDTH","BOX_HEIGHT"} <= xy),
        pedestrian_availability=availability, frame_and_geometry_alignment_verified=False,
        coordinate_conversion_verified=False, paired_targets_for_evaluation_released=False,
        new_role_assigned=False, forecast_errors_evaluated=False)


if __name__ == "__main__":
    PUBLIC.mkdir(parents=True,exist_ok=True)
    immutable(PUBLIC/"paired_source_preflight.json",run())
