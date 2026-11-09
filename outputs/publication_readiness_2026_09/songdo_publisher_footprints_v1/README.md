# Songdo: screen candidate locations before opening trajectories

The [publisher's Songdo v2 record](https://zenodo.org/records/17924857)
lists 20 intersections across four days, with 80 trajectory archives
(12,205,648,078 bytes). My local first-day sample covers A, B, C and E.
The other 16 first-day archives total 2,321,671,587 bytes. Dates are
repeats of locations, not separate geographic sources.

I read the 20 orthophoto center text members using 565,603 bytes of
HTTP range payload, without downloading the full 1.78 GB image archive.
Each returned range and ZIP member CRC was checked; the full archive
checksum was not computed. The published 15,000-pixel base-resolution
cutouts give three overlapping pairs, **A/B, F/G and K/L**. The
conservative rectangle screen therefore yields **17 map components**.
It independently flags the A/B link found in the earlier observed
trajectory footprint audit.

These are map components, not verified independent trajectory groups:
disjoint images do not exclude a vehicle crossing sites or a related
capture. A later approved expansion should check timestamps and source
links before assigning groups. No trajectory row, forecast label,
orthophoto image member or official score was read. No split was
assigned and the existing external results are unchanged.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-pytorch/bin/python scripts/audit_m3w_songdo_publisher_footprints.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_songdo_publisher_footprints.py
```

[Publisher file counts, center-member hashes and component screen](readout.json).
