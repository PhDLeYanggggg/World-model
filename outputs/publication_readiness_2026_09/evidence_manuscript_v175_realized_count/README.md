# v175: OpenDD with the realized edit count fixed

I added the [fixed-count comparison](../opendd_realized_stream_capacity_v1/README.md)
to the OpenDD section. It distinguishes the original shared ceiling (330/606
hindsight versus 57/606 online) from a fixed-count comparison (242/606 versus
57/606). The latter matches the actual number of edits in each recording and
seed, not total displacement. It remains a development diagnostic.

```sh
python scripts/build_m3w_verified_references_candidate.py --realized-count --compiler /path/to/tectonic
```

Only the manuscript source changes in the archive; the title, abstract,
disclosure, references, earlier experiments and figure assets stay unchanged.
The five-page PDF and source archive remain local. The anonymous v3 supplement
still replays the original results; it does not replay this additional analysis.
See the [build receipt](receipt.json). Nothing was uploaded.
