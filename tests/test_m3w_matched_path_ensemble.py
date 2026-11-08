from scripts.run_m3w_matched_path_ensemble import base, source_members


def test_source_members_reuse_refiner_and_exclude_source_from_every_producer():
    held, seed = base.SITES[1], base.SEEDS[0]
    refiner = object()
    inner = {(tuple(sorted((held, peer))), seed): peer for peer in base.SITES if peer != held}
    members = source_members(held, seed, ("site_excluded", refiner), inner)
    assert [producer for producer, _ in members] == ["site_excluded", *[s for s in base.SITES if s != held]]
    assert all(head is refiner for _, head in members)
