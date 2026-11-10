import numpy as np
import pytest

from scripts import audit_m3w_paper_evidence_lineage as lineage


def test_roster_alignment_and_actual_overlap_boundary():
    names = np.array(['same', 'same', 'other'])
    offsets = np.array([0, 2, 5, 8])
    lineage.cohort_layout(names, np.array([0, 240, 0]), offsets, 8, 12)
    with pytest.raises(ValueError, match='overlapping'):
        lineage.cohort_layout(names, np.array([0, 228, 0]), offsets, 8, 12)
    with pytest.raises(ValueError, match='indexing'):
        lineage.cohort_layout(names, np.array([0, 240, 0]), np.array([0, 5, 2, 8]), 8, 12)


def test_scope_requires_all_and_only_donor_sources():
    lineage.fitting_scope({'fit': ['c']}, {'a', 'b', 'c'}, ['a', 'b'])
    for wrong in (['a', 'c'], ['c', 'c'], []):
        with pytest.raises(ValueError, match='wrong fitting source'):
            lineage.fitting_scope({'fit': wrong}, {'a', 'b', 'c'}, ['a', 'b'])
