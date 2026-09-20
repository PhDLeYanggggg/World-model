"""Persist changed heartbeat observations for an existing PID; never restart it."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--family',choices=('transformer','eqmotion'),required=True)
    parser.add_argument('--pid',type=int,required=True)
    args=parser.parse_args()
    folder=ROOT/'data/stage_cvpr2027_experiments/frozen_interaction_v1'/args.family
    last=None
    while True:
        status=json.loads((folder/'heartbeat.json').read_text())
        if status['pid']!=args.pid:raise RuntimeError('Observed PID changed; do not follow a different job silently')
        try:
            os.kill(args.pid,0);alive=True
        except ProcessLookupError:alive=False
        except PermissionError:alive=None
        value=dict(observed_utc=datetime.now(timezone.utc).isoformat(),process_exists=alive,heartbeat=status)
        if status!=last or alive is False:
            with (folder/'observed_progress.jsonl').open('a') as stream:
                stream.write(json.dumps(value,allow_nan=False)+'\n')
            print(json.dumps(value),flush=True);last=status
        if status['status']=='complete_not_selected':return
        if alive is False:raise RuntimeError('Observed process terminal without a completed heartbeat; do not restart automatically')
        time.sleep(10)


if __name__=='__main__':main()
