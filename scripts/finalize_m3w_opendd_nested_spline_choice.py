"""Finish v2 using its unchanged registered training code and saved checkpoints."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_nested_spline_choice as m


def portable_rules(rules):
    # NumPy scalar booleans are not supported by the strict receipt writer.
    return {k: (rule, [dict(row, admissible=bool(row['admissible'])) for row in table])
            for k, (rule, table) in rules.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', action='store_true', help='Require existing complete checkpoints')
    args = p.parse_args()
    run, torch = m.run, m.torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    m.PRIVATE.mkdir(parents=True, exist_ok=True)
    m.control.two.fit.PRIVATE = m.PRIVATE
    start = time.monotonic()
    with (m.PRIVATE/'train.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            z = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            run.kin.prior.source.atomic(m.PRIVATE/'heartbeat.json', z)
            if v.get('phase') != 'fit':
                print(z, flush=True)

        reg, source, freeze, slots, ph, h, anchor = m.register()
        repair = m.OUT/'receipt_repair_v2.json'
        run.once(repair, run.stable(dict(registration=reg, code=run.artifact(Path(__file__)),
            change='cast admissibility flags to JSON-native bool; no scoring or training changes',
            prior_failed_before_freeze=True, original_2pct=True, confirmation_access=False)))
        data = m.m.load(source)
        cells, cal, fits, heads, updates = [], {}, {}, {}, {}
        for held, block, rules, fits, heads, updates in m.fit(
                data, source, reg, freeze, ph, h, anchor, args.replay, beat):
            cells.extend(block)
            cal[held] = portable_rules(rules)
        fp = m.OUT/'freeze.json'
        run.once(fp, run.stable(dict(registration=reg, receipt_repair=run.sha(repair),
            cells=cells, calibration=cal, producer_fits=fits, selector_fits=heads,
            before_held_scoring=True,
            configured_updates=dict(producer=m.m.STEPS*len(fits), selector=m.STEPS*len(heads)))))
        rows, summary = m.evaluate(data, run.read(fp)['cells'], slots)
        run.once(m.OUT/'readout.json', run.stable(dict(freeze=run.sha(fp), summary=summary, rows=rows,
            original_2pct=True, confirmation_access=False, independent_confirmation=False, deployment_changed=False)))
        beat(phase='complete', updates=dict(updates), replay=args.replay)


if __name__ == '__main__':
    main()
