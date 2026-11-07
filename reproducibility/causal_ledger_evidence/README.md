# Reproduce The Ledger Tables

Extract the review bundle and run:

```sh
python3 -I -S reproduce.py --verify --output reproduced
```

The standalone script uses only the Python standard library. It checks
hashes and reconstructs the matched contrasts from paired source/seed
masses, using the original 3,000 source bootstrap draws. It recalculates
all archived policy risk pass counts from the 102 repeated recording
views, keeping undefined risk unsupported. It also sums the predicted
credit and the realized signed excess on those same selected queries.

Outputs: `paired_comparisons.csv`, `recording_risks.csv`, `replayed.json`.
Use a new output directory for each run. Nothing is downloaded.

The package contains aggregate evidence from already explored source
groups, not independent confirmation. It has no raw tracks, images,
identities, model weights or fitted coefficients. It does not reproduce
forecast training, inference or the construction of the input masses.
Direct identifiers are removed, but numerical results can be recognizable
from public reports. This is a draft reproduction aid, not a submission
or an anonymity guarantee. The error unit is image coordinates, with eight
observations and twelve prediction steps, not calibrated distance or time.
