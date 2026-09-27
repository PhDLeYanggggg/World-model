"""Serialization-only adapter for the immutable registered report implementation."""
import copy
from numbers import Integral
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import report_m3w_european_cost_mass as registered

COUNT = 'L2_MSE_improves_but_mass_log_error_worsens'


def native_counts(summary):
    result = copy.deepcopy(summary)
    for arms in result.values():
        for arm in arms.values():
            for weighting in ('equal_locality', 'row_weighted'):
                for component in ('H_all', 'H_easy'):
                    row = arm[weighting][component]
                    value = row[COUNT]
                    if not isinstance(value, Integral):
                        raise TypeError('Diagnostic count must already be integral')
                    row[COUNT] = int(value)
                    assert row[COUNT] == value
    return result


def main():
    original = registered.fitting_summary
    # Preserve hash-bound scientific code; change only its JSON boundary type.
    registered.fitting_summary = lambda records: native_counts(original(records))
    try:
        registered.main()
    finally:
        registered.fitting_summary = original


if __name__ == '__main__': main()
