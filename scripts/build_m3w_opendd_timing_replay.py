"""Package only anonymous scalar evidence for the recent timing experiments."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_opendd_candidate_replay as legacy
from src.reproduction import m3w_opendd_timing_replay as replay

OUT = legacy.EXTENDED.parent.parent/"opendd_timing_replay_v1"/"anonymous_opendd_timing.zip"
EXTENDED = OUT.parent.parent/"opendd_timing_replay_v2"/OUT.name
RECORDING = OUT.parent.parent/"opendd_timing_replay_v3"/OUT.name
RESERVED = OUT.parent.parent/"opendd_timing_replay_v4"/OUT.name
SHAPES = OUT.parent.parent/"opendd_timing_replay_v5"/OUT.name
TWO_SLOTS = OUT.parent.parent/"opendd_timing_replay_v6"/OUT.name
STOPPING = OUT.parent.parent/"opendd_timing_replay_v7"/OUT.name
ADMISSION = OUT.parent.parent/"opendd_timing_replay_v8"/OUT.name
RECORD_SUPPORT = OUT.parent.parent/"opendd_timing_replay_v9"/OUT.name
SCHEDULE_SUPPORT = OUT.parent.parent/"opendd_timing_replay_v10"/OUT.name
DUAL_DIRECTION = OUT.parent.parent/"opendd_timing_replay_v11"/OUT.name
DUAL_CLOCK = OUT.parent.parent/"opendd_timing_replay_v12"/OUT.name
BUDGET_RANK = OUT.parent.parent/"opendd_timing_replay_v13"/OUT.name
PROTECTED_GAIN_REPLAY = OUT.parent.parent/"opendd_timing_replay_v14"/OUT.name
GAIN_WAIT_REPLAY = OUT.parent.parent/"opendd_timing_replay_v15"/OUT.name
SLATE = "opendd_slate_queue_v1"
RANK = "opendd_slate_rank_v1"
TEACHER = "opendd_record_teacher_v1"
RESERVE = "opendd_reserve_v1"
FACTOR = "opendd_factorized_reserve_v1"
SHAPE = "opendd_reserve_shape_v1"
SINGLE = "opendd_reserve_only_v1"
TWO = "opendd_two_reserves_v1"
TWO_RANK = "opendd_two_reserve_ranking_v1"
STOP = "opendd_support_stop_v1"
ADMIT = "opendd_support_admission_v1"
RECORD = "opendd_record_support_v1"
MONOTONE = "opendd_support_monotonicity_v1"
SCHEDULE = "opendd_support_schedule_v1"
DIRECTION = "opendd_dual_direction_v1"
CLOCK = "opendd_dual_clock_v1"
DEADLINE = CLOCK+"_deadline"
RANK_BUDGET = "opendd_budget_rank_v1"
ATTRIBUTION = RANK_BUDGET+"_attribution"
PROTECTED_GAIN = "opendd_protected_gain_v1"
GAIN_WAIT = "opendd_gain_wait_v1"
SLATE_ORDER = ("adaptive_queue_vs_cv", "adaptive_queue_vs_same_clock",
               "adaptive_queue_vs_same_credits_immediate", "adaptive_queue_vs_fixed_queue",
               "adaptive_queue_vs_no_geometry", "adaptive_queue_vs_myopic_queue",
               "adaptive_now_vs_fixed_now", "pair_relative")


def read_clock():
    p = legacy.REPORTS/SLATE
    r = json.loads((p/"clock_registration.json").read_text())
    a = json.loads((p/"clock_capacity.json").read_text())
    if (a["registration"] != legacy.sha(p/"clock_registration.json") or not r["preserves_recording_query_counts"]
            or r["parent"]["sha256"] != legacy.sha(p/"analysis.json")
            or any(x["confirmation_access"] or not x["original_2pct"] for x in (r, a))):
        raise ValueError("Require frozen outcome-aware clock diagnostic")
    return a


def read_support():
    p = legacy.REPORTS/TEACHER
    r = json.loads((p/"support_registration.json").read_text())
    a = json.loads((p/"support_capacity.json").read_text())
    if (a["registration"] != legacy.sha(p/"support_registration.json") or not r["outcome_aware_only"]
            or not a["outcome_aware_only"] or r["changes_main_budget"]
            or r["parents"][0]["sha256"] != legacy.sha(p/"analysis.json")
            or any(z["confirmation_access"] or not z["original_2pct"] for z in (r, a))):
        raise ValueError("Require the frozen diagnostic-only support capacity")
    return a


def read_factor():
    p = legacy.REPORTS/RESERVE
    r = json.loads((p/"factor_registration.json").read_text())
    a = json.loads((p/"factor_analysis.json").read_text())
    if (a["registration"] != legacy.sha(p/"factor_registration.json")
            or r["parent"]["sha256"] != legacy.sha(p/"analysis.json") or not r["no_new_thresholds"]
            or any(z["confirmation_access"] or not z["original_2pct"] for z in (r, a))):
        raise ValueError("Require frozen factorized reserve evidence")
    return a


def shape_tables(studies, expected):
    a = legacy.read(SHAPE)
    for clock in ("gain", "easy", "incumbent"):
        key = f"{SHAPE}_{clock}"
        s = a["summary"][clock]
        count = {k: {f: v[f] for f in legacy.FIELDS} for k, v in s["metrics"].items()}
        x = [[dict(effects=r[clock]["effects"],
                   counts={k: {f: r[clock]["metrics"][k][f] for f in v} for k, v in count.items()})
              for r in sorted(z, key=lambda x: x["seed"])] for _, z in sorted(a["by_site"].items())]
        studies[key] = dict(sites=x, effect_order=sorted(s["effects"]), sequential_draws=False)
        expected[key] = dict(counts=count, effects={k: {f: v[f] for f in ("point", "interval95")}
                                                   for k, v in s["effects"].items()})


def read_two(suffix, flag):
    p = legacy.REPORTS/TWO
    f = p/f"{suffix}_registration.json"
    r = json.loads(f.read_text())
    a = json.loads((p/("ranking_analysis.json" if suffix == "ranking" else "timing_capacity.json")).read_text())
    if (a["registration"] != legacy.sha(f) or r["parent"]["sha256"] != legacy.sha(p/"analysis.json")
            or not r[flag] or any(x["confirmation_access"] or not x["original_2pct"] for x in (r, a))):
        raise ValueError("Require frozen two-slot follow-up")
    return a


def read_menu():
    p = legacy.REPORTS/STOP
    f = p/"menu_registration.json"
    r = json.loads(f.read_text())
    a = json.loads((p/"menu_capacity.json").read_text())
    if (a["registration"] != legacy.sha(f) or r["parent"]["sha256"] != legacy.sha(p/"analysis.json")
            or any(x["confirmation_access"] or not x["original_2pct"] or not x["outcome_aware_only"]
                   for x in (r, a))):
        raise ValueError("Require frozen development-only recipient menu")
    return a


def read_transitions():
    p = legacy.REPORTS/MONOTONE
    f = p/"registration.json"
    r = json.loads(f.read_text())
    a = json.loads((p/"analysis.json").read_text())
    if (a["registration"] != legacy.sha(f)
            or r["parent"]["sha256"] != legacy.sha(legacy.REPORTS/ADMIT/"analysis.json")
            or not r["undefined_kept"]
            or any(x["confirmation_access"] or not x["original_2pct"] for x in (r, a))):
        raise ValueError("Require frozen risk transition evidence")
    return a


def transition_tables(studies, expected):
    a = read_transitions()
    s = a["summary"]
    count = dict(transitions=s["transitions"], **s["numeric"])
    sites = []
    for _, rows in sorted(a["by_site"].items()):
        group = []
        for r in sorted(rows, key=lambda z: z["seed"]):
            local = dict(transitions=r["transitions"], **r["numeric"])
            group.append(dict(counts={k: {f: local[k].get(f, 0) for f in keys} for k, keys in count.items()},
                              effects={}))
        sites.append(group)
    studies[MONOTONE] = dict(sites=sites, effect_order=[], sequential_draws=False)
    expected[MONOTONE] = dict(counts=count, effects={})


def read_schedule_audit(name, registration):
    p = legacy.REPORTS/SCHEDULE
    f = p/f"{registration}.json"
    r = json.loads(f.read_text())
    a = json.loads((p/f"{name}.json").read_text())
    lineage = r.get("parents", [r.get("parent")])
    root = legacy.REPORTS.parents[1]
    if (a["registration"] != legacy.sha(f) or lineage[0]["sha256"] != legacy.sha(p/"analysis.json")
            or any(legacy.sha(root/x["path"]) != x["sha256"] for x in lineage)
            or any(x["confirmation_access"] or not x["original_2pct"] or not x["outcome_aware_only"]
                   for x in (r, a))):
        raise ValueError("Require verified schedule capacity evidence")
    return a


def schedule_tables(studies, expected):
    a = legacy.read(SCHEDULE)
    b = read_schedule_audit("capacity_breakdown", "capacity_registration")
    c = read_schedule_audit("shape_capacity", "shape_registration")
    counts = dict(timing=a["summary"]["capacity"], **b["summary"], shapes=c["summary"])
    groups = []
    for site, rows in sorted(a["by_site"].items()):
        sites = []
        for r in sorted(rows, key=lambda z: z["seed"]):
            lookup = lambda doc: next(z for z in doc["by_site"][site] if z["seed"] == r["seed"])
            sources = dict(timing=r["capacity"], shapes=lookup(c)["counts"])
            sources.update({k: v for k, v in lookup(b).items() if k in ("counts", "reasons")})
            sites.append(dict(counts={k: {f: sources[k].get(f, 0) for f in keys} for k, keys in counts.items()},
                              effects={}))
        groups.append(sites)
    key = SCHEDULE+"_exact"
    studies[key] = dict(sites=groups, effect_order=[], sequential_draws=False)
    expected[key] = dict(counts=counts, effects={})


def read_deadline():
    p = legacy.REPORTS/CLOCK
    r = json.loads((p/"deadline_registration.json").read_text())
    a = json.loads((p/"deadline_analysis.json").read_text())
    root = legacy.REPORTS.parents[1]
    if (a["registration"] != legacy.sha(p/"deadline_registration.json") or not r["no_new_training"]
            or any(legacy.sha(root/z["path"]) != z["sha256"] for z in r["parents"])
            or any(z["confirmation_access"] or not z["original_2pct"] for z in (r, a))):
        raise ValueError("Require the verified no-refit deadline diagnostic")
    return a


def read_attribution():
    p = legacy.REPORTS/RANK_BUDGET
    r = json.loads((p/"attribution_registration.json").read_text())
    a = json.loads((p/"attribution_analysis.json").read_text())
    root = legacy.REPORTS.parents[1]
    if (a["registration"] != legacy.sha(p/"attribution_registration.json") or not r["no_new_training"]
            or any(legacy.sha(root/z["path"]) != z["sha256"] for z in r["parents"])
            or any(z["confirmation_access"] or not z["original_2pct"] for z in (r, a))):
        raise ValueError("Require the verified no-refit swap attribution")
    return a


def read_protected_clock():
    p = legacy.REPORTS/PROTECTED_GAIN
    r = json.loads((p/"clock_registration.json").read_text())
    a = json.loads((p/"clock_capacity.json").read_text())
    root = legacy.REPORTS.parents[1]
    if (a["registration"] != legacy.sha(p/"clock_registration.json")
            or any(legacy.sha(root/z["path"]) != z["sha256"] for z in r["parents"])
            or any(z["confirmation_access"] or not z["original_2pct"] or not z["outcome_aware_only"]
                   for z in (r, a))):
        raise ValueError("Require the verified outcome-aware protected gain clock")
    return a


def read_gain_wait_menu():
    p = legacy.REPORTS/GAIN_WAIT
    r = json.loads((p/"menu_registration.json").read_text())
    a = json.loads((p/"menu_capacity.json").read_text())
    root = legacy.REPORTS.parents[1]
    if (a["registration"] != legacy.sha(p/"menu_registration.json")
            or any(legacy.sha(root/z["path"]) != z["sha256"] for z in r["parents"])
            or any(z["confirmation_access"] or not z["original_2pct"] or not z["outcome_aware_only"]
                   for z in (r, a))):
        raise ValueError("Require the frozen gain-wait menu attribution")
    return a


def capacity_table(studies, expected, a, key, field):
    keys = set(a["summary"]).union(*(r["counts"] for rows in a["by_site"].values() for r in rows))
    groups = [[dict(counts={field: {k: r["counts"].get(k, 0) for k in keys}}, effects={})
               for r in sorted(rows, key=lambda z: z["seed"])]
              for _, rows in sorted(a["by_site"].items())]
    studies[key] = dict(sites=groups, effect_order=[], sequential_draws=False)
    expected[key] = dict(counts={field: {k: a["summary"].get(k, 0) for k in keys}}, effects={})


def build(include_rank=False, include_teacher=False, include_reserve=False, include_shape=False,
          include_two=False, include_stop=False, include_admission=False, include_record=False,
          include_schedule=False, include_direction=False, include_clock=False, include_budget_rank=False,
          include_protected_gain=False, include_gain_wait=False):
    studies, expected = {}, {}
    include_protected_gain = include_protected_gain or include_gain_wait
    include_budget_rank = include_budget_rank or include_protected_gain
    include_clock = include_clock or include_budget_rank
    include_direction = include_direction or include_clock
    include_schedule = include_schedule or include_direction
    include_record = include_record or include_schedule
    include_admission = include_admission or include_record
    include_stop = include_stop or include_admission
    include_two = include_two or include_stop
    include_shape = include_shape or include_two
    include_reserve = include_reserve or include_shape
    names = ("opendd_release_credit_v1", "opendd_credit_value_v2", SLATE)
    if include_rank or include_teacher or include_reserve:
        names += (RANK,)
    if include_teacher or include_reserve:
        names += (TEACHER,)
    if include_reserve:
        names += (RESERVE, FACTOR)
    if include_shape:
        names += (SINGLE,)
    if include_two:
        names += (TWO, TWO_RANK)
    if include_stop:
        names += (STOP,)
    if include_admission:
        names += (ADMIT,)
    if include_record:
        names += (RECORD,)
    if include_schedule:
        names += (SCHEDULE,)
    if include_direction:
        names += (DIRECTION,)
    if include_clock:
        names += (CLOCK, DEADLINE)
    if include_budget_rank:
        names += (RANK_BUDGET, ATTRIBUTION)
    if include_protected_gain:
        names += (PROTECTED_GAIN,)
    if include_gain_wait:
        names += (GAIN_WAIT,)
    for name in names:
        a = (read_factor() if name == FACTOR else read_two("ranking", "no_threshold_search") if name == TWO_RANK
             else read_deadline() if name == DEADLINE
             else read_attribution() if name == ATTRIBUTION else legacy.read(name))
        s = a["summary"]
        fields = {k: {f: v[f] for f in legacy.FIELDS} for k, v in s["metrics"].items()}
        status = {} if name == SCHEDULE else s.get("capacity", {})
        clock = read_clock() if name == SLATE else None
        lookup = {(k, r["seed"]): r["capacity"] for k, rows in clock["by_site"].items() for r in rows} if clock else {}
        support = read_support() if name == TEACHER else None
        support_rows = {(k, r["seed"]): r["groups"] for k, rows in support["by_site"].items() for r in rows} if support else {}
        timing = read_two("timing", "outcome_aware_only") if name == TWO else None
        timing_rows = {(k, r["seed"]): r["counts"] for k, rows in timing["by_site"].items() for r in rows} if timing else {}
        menu = read_menu() if name == STOP else None
        menu_rows = {(k, r["seed"]): r["counts"] for k, rows in menu["by_site"].items() for r in rows} if menu else {}
        scalars = dict(s["effects"])
        if name == SLATE:
            scalars["pair_relative"] = s["pair_relative"]
        cells = []
        for site, rows in sorted(a["by_site"].items()):
            view = []
            for r in sorted(rows, key=lambda v: v["seed"]):
                effect = dict(r["effects"])
                if name == SLATE:
                    effect["pair_relative"] = r["pair_relative"]
                counts = {k: {f: r["metrics"][k][f] for f in v} for k, v in fields.items()}
                for delay, table in status.items():
                    counts[f"capacity_{delay}"] = {k: r["capacity"][delay].get(k, 0) for k in table}
                if clock:
                    counts["same_clock_capacity"] = {k: lookup[site, r["seed"]].get(k, 0) for k in clock["summary"]}
                if support:
                    counts.update({f"extra_edit_{group}": {k: support_rows[site, r["seed"]][group].get(k, 0)
                                  for k in values} for group, values in support["summary"].items()})
                if timing:
                    counts["second_slot_timing"] = {k: timing_rows[site, r["seed"]].get(k, 0) for k in timing["summary"]}
                if menu:
                    counts.update({f"recipient_menu_{k}": v for k, v in menu_rows[site, r["seed"]].items()})
                view.append(dict(effects=effect, counts=counts))
            cells.append(view)
        order = list(SLATE_ORDER if name == SLATE else sorted(scalars))
        if set(order) != set(scalars):
            raise ValueError("Unexpected contrast order")
        studies[name] = dict(sites=cells, effect_order=order, sequential_draws=name == SLATE)
        fields.update({f"capacity_{k}": v for k, v in status.items()})
        if clock:
            fields["same_clock_capacity"] = clock["summary"]
        if support:
            fields.update({f"extra_edit_{k}": value for k, value in support["summary"].items()})
        if timing:
            fields["second_slot_timing"] = timing["summary"]
        if menu:
            fields.update({f"recipient_menu_{k}": v for k, v in menu["summary"].items()})
        expected[name] = dict(counts=fields, effects={k: {f: v[f] for f in ("point", "interval95")} for k, v in scalars.items()})
    if include_shape:
        shape_tables(studies, expected)
    if include_record:
        transition_tables(studies, expected)
    if include_schedule:
        schedule_tables(studies, expected)
    if include_protected_gain:
        capacity_table(studies, expected, read_protected_clock(), PROTECTED_GAIN+"_clock", "clock_capacity")
    if include_gain_wait:
        capacity_table(studies, expected, read_gain_wait_menu(), GAIN_WAIT+"_menu", "menu_capacity")
    replay.verify(replay.aggregate(studies), expected)
    files = {
        "replay.py": Path(replay.__file__).read_bytes(),
        "requirements.txt": f"numpy=={np.__version__}\n".encode(),
        "README.md": b"# OpenDD timing evidence\n\nRun python replay.py with the pinned NumPy version.\n"
                     b"These anonymous site/seed scalars reproduce summaries and intervals, not training.\n"
                     b"The same-clock optimum and release ceilings are outcome-aware diagnostics.\n"
                     b"None of the results is independent confirmation or deployment evidence.\n",
        "statistics.json": legacy.encoded(dict(studies=studies, independent_confirmation=False, original_risk_cap=.02)),
        "expected.json": legacy.encoded(expected),
    }
    files["manifest.json"] = legacy.encoded({k: hashlib.sha256(v).hexdigest() for k, v in files.items()})
    out = next(p for enabled, p in ((include_gain_wait, GAIN_WAIT_REPLAY), (include_protected_gain, PROTECTED_GAIN_REPLAY),
        (include_budget_rank, BUDGET_RANK), (include_clock, DUAL_CLOCK),
        (include_direction, DUAL_DIRECTION),
        (include_schedule, SCHEDULE_SUPPORT), (include_record, RECORD_SUPPORT),
        (include_admission, ADMISSION), (include_stop, STOPPING),
        (include_two, TWO_SLOTS), (include_shape, SHAPES), (include_reserve, RESERVED),
        (include_teacher, RECORDING), (include_rank, EXTENDED), (True, OUT)) if enabled)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".pending")
    with ZipFile(tmp, "w") as z:
        for k, v in sorted(files.items()):
            item = ZipInfo(k, date_time=(1980, 1, 1, 0, 0, 0))
            item.external_attr = 0o100644 << 16
            z.writestr(item, v, compress_type=ZIP_DEFLATED, compresslevel=9)
    tmp.replace(out)
    print(out, legacy.sha(out), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--include-rank", action="store_true")
    p.add_argument("--include-record-teacher", action="store_true")
    p.add_argument("--include-reserve", action="store_true")
    p.add_argument("--include-reserve-shape", action="store_true")
    p.add_argument("--include-two-reserves", action="store_true")
    p.add_argument("--include-support-stop", action="store_true")
    p.add_argument("--include-support-admission", action="store_true")
    p.add_argument("--include-record-support", action="store_true")
    p.add_argument("--include-support-schedule", action="store_true")
    p.add_argument("--include-dual-direction", action="store_true")
    p.add_argument("--include-dual-clock", action="store_true")
    p.add_argument("--include-budget-rank", action="store_true")
    p.add_argument("--include-protected-gain", action="store_true")
    p.add_argument("--include-gain-wait", action="store_true")
    a = p.parse_args()
    build(a.include_rank, a.include_record_teacher, a.include_reserve, a.include_reserve_shape,
          a.include_two_reserves, a.include_support_stop, a.include_support_admission,
          a.include_record_support, a.include_support_schedule, a.include_dual_direction, a.include_dual_clock,
          a.include_budget_rank, a.include_protected_gain, a.include_gain_wait)
