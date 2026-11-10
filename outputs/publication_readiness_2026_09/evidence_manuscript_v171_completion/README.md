# Moving completion bound in the manuscript

This separate eight-page candidate adds the observed-step and favorable-label
limits next to the original conservative result. The first six rendered pages
are pixel-identical to v170. Pages seven and eight reflow around the changed
paragraph. The abstract, figures, tables, references and LLM statement are
unchanged. No portal action was taken.

```sh
python scripts/build_m3w_completion_candidate.py --compiler /path/to/tectonic
```

The builder needs pypdf, the verified v170 source and the development
[completion receipt](../moving_completion_envelope_v1/receipt.json). It writes
a separate local PDF and source archive, and checks eight pages, the rendered
results and the unchanged disclosure. [Build receipt](receipt.json).

The Songdo v2 review archive remains unchanged. Its offline replay verifies
the earlier result; it does not recompute this additional Moving bound.
