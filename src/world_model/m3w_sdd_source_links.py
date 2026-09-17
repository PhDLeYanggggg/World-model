"""Explicit diagnostic media-link hypothesis; never rename source files."""
from __future__ import annotations


def lexical_reindex_links(annotation_names):
    names = list(annotation_names)
    if not names or len(names) != len(set(names)):
        raise ValueError('Nonempty unique annotation video names required')
    if set(names) != {'video'+str(i) for i in range(len(names))}:
        raise ValueError('Require complete numbered source inventory')
    return {name: 'video'+str(index) for index, name in enumerate(sorted(names))}
