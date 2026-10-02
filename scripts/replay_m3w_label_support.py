"""Rerun the frozen diagnostic without replacing its immutable runtime receipt."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VOLATILE = {'pid','seconds','peak_RSS_bytes','available_disk_bytes'}


def verify_receipt(old, new):
    if old.keys() != new.keys(): raise AssertionError('Runtime receipt schema changed')
    for key in old.keys()-VOLATILE:
        if old[key] != new[key]: raise AssertionError('Replay evidence changed: '+key)


def main():
    from scripts import run_m3w_label_support as frozen
    original = frozen.once
    destination = frozen.PUBLIC/'replay.json'
    assert destination.is_file(), 'Use the original verify phase for the first replay'

    def checked_writer(path,value):
        if path == destination:
            verify_receipt(json.loads(path.read_text()),value)
            new_path=frozen.PRIVATE/'additional_replays'/(str(time.time_ns())+'.json')
            original(new_path,value)
            print(json.dumps(dict(read_only_scientific_replay=True,runtime_receipt=str(new_path))),flush=True)
        else:
            # Every scientific artifact still uses the frozen exact comparison.
            original(path,value)

    frozen.once=checked_writer
    sys.argv=[sys.argv[0],'verify']
    frozen.main()


if __name__ == '__main__': main()
