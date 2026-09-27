# Trained Association Response

First256 eligible held-source histories per fixed model, selected only by observed masks.
Swap two complete neighbor identities at past slots1/3/5, holding each time point set,
ego, timestamps, current positions and masks unchanged. No future labels are read.
The constructed paths need not be physically plausible. Sensitivity is not accuracy.

| Trial | Held queries | Flat mean output change | Grouped mean output change | Flat invariant | Grouped invariant |
|---|---:|---:|---:|---|---|
| single0_seed17 | 256 | 5.9073608e-07 | 0.0065580406 | True | False |
| single0_seed29 | 256 | 6.1534439e-07 | 0.0056072022 | True | False |
| single0_seed43 | 256 | 5.1769007e-07 | 0.0065545365 | True | False |
| single1_seed17 | 256 | 3.4053585e-06 | 0.30753237 | True | False |
| single1_seed29 | 256 | 3.9693309e-06 | 0.45147434 | True | False |
| single1_seed43 | 256 | 4.1693579e-06 | 0.57782054 | False | False |
| single2_seed17 | 256 | 7.8930651e-07 | 0.0039619552 | True | False |
| single2_seed29 | 256 | 9.0865024e-07 | 0.011252235 | True | False |
| single2_seed43 | 256 | 8.5155654e-07 | 0.0075277998 | True | False |

These queries and fitted models overlap; this is not an independent statistical test.
The probe checks the intended mechanism but does not establish interaction benefit.
