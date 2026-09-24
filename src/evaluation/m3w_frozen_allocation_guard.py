"""Fail closed on source or runtime identity drift in a frozen repair replay."""


def require_frozen_repair_identity(current, frozen_parent, frozen_repair):
    actual = current['source_bindings']
    for path, expected in frozen_parent['source_bindings'].items():
        if actual.get(path) != expected:
            raise ValueError('Frozen parent source mismatch: '+path)
    if current != frozen_repair:
        raise ValueError('Reconstructed repair identity differs from frozen repair identity')

