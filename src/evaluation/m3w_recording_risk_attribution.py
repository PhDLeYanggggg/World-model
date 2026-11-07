"""Explain a frozen recording verdict without changing its risk rule."""
from collections import Counter


def attribute(record):
    pairs = (
        (record['known_all_risk'], record['completion_all_risk_upper']),
        (record['known_easy_risk'], record['completion_easy_risk_upper']),
    )
    signed = record['known_easy_net_degradation']
    if any((a is None) != (b is None) or (a is not None and a > b) for a, b in pairs):
        raise ValueError('Inconsistent observed and completion risk')
    all_check, easy_check = (
        'undefined' if a is None else 'observed_exceeds' if a > .02 else
        'debit_exceeds' if b > .02 else 'within'
        for a, b in pairs
    )
    signed_check = 'undefined' if signed is None else 'observed_exceeds' if signed > .02 else 'within'
    checks = (all_check, easy_check, signed_check)
    if record['supported'] != all(c == 'within' for c in checks):
        raise ValueError('Attribution disagrees with the original recording verdict')
    return checks


def summarize(records):
    checks = [attribute(r) for r in records]
    states = ('within', 'observed_exceeds', 'debit_exceeds', 'undefined')
    summary = {
        key: {state: tally[state] for state in states}
        for key, tally in zip(('all_positive_harm', 'easy_positive_harm', 'signed_easy'),
                              (Counter(column) for column in zip(*checks, strict=True)))
    }
    overlaps = Counter()
    for c in checks:
        reasons = [state for state in states[1:] if state in c]
        overlaps['+'.join(reasons) if reasons else 'pass'] += 1
    return dict(checks=summary, disjoint_recording_patterns=dict(overlaps),
                views=len(checks), switches=sum(r['selected'] for r in records),
                unknown_switches=sum(r['unknown_selected'] for r in records))
