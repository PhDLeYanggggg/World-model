# Students01 Upstream Packaging Audit

Fresh source-to-source row mapping; no neural scores or threshold search.

The continuous file has 21813 rows / 415 identities; the packaged file has 17820 rows / 891 identities.
Every packaged position uniquely matches the same native frame and the continuous coordinate rounded to three decimals.
There is no detected frame reset or coordinate mismatch. This is identity fragmentation and availability conditioning, not arbitrary fabricated clocks.

The package exactly retains floor(track_length/20)*20 points: True.
It removes 3993 tail/short-track rows and all 63 tracks shorter than 20 points.
A new ID starts each 20-point chunk. Determining whether to retain a chunk uses its later availability.
Therefore a reader that uses only past rows cannot on its own prove causal eligibility of the upstream context population.
This is distinct from passing a future endpoint to the neural network or leaking a train/test duplicate.

| Source | Past-supported queries | Complete obs8/pred12 labels |
| --- | ---: | ---: |
| continuous | 18920 | 14295 |
| packaged | 11583 | 891 |

## Consequence

The v1-v4 Students01 results use this packaged observation universe and cannot certify full-scene causal deployment.
Training recordings are unchanged by this finding. Preserve completed/partial fits and historical results, but do not continue toward a positive claim using the affected evaluation context.
Repair in a new version: use the continuous local source with original IDs and precision, retain short/past-only agents for context, and read future availability only for scoring.
Do not rewrite existing cache files, protocol hashes or checkpoint identities. Historical exposure remains development-only; the repair does not create an untouched test set.
The continuous file is still an annotation product: this audit does not verify its upstream interpolation, physical calibration, seconds, or sensor-as-of causality.
