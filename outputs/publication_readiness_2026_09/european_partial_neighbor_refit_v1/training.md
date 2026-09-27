# Training Endpoints

| Trial | Parameters | Updates | Draws | Unique train queries | Held draws | Fit seconds | Last logged loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| single0_seed17 | 88514 | 4000 | 256000 | 60181 | 0 | 94.459 | 0.893336 |
| single0_seed29 | 88514 | 4000 | 256000 | 60531 | 0 | 98.631 | 1.045282 |
| single0_seed43 | 88514 | 4000 | 256000 | 60331 | 0 | 99.265 | 1.243410 |
| single1_seed17 | 88514 | 4000 | 256000 | 23754 | 0 | 99.647 | 1.154015 |
| single1_seed29 | 88514 | 4000 | 256000 | 23755 | 0 | 101.089 | 1.138660 |
| single1_seed43 | 88514 | 4000 | 256000 | 23758 | 0 | 100.983 | 0.767034 |
| single2_seed17 | 88514 | 4000 | 256000 | 69367 | 0 | 102.614 | 0.609232 |
| single2_seed29 | 88514 | 4000 | 256000 | 69873 | 0 | 102.879 | 1.116618 |
| single2_seed43 | 88514 | 4000 | 256000 | 69562 | 0 | 100.363 | 0.788563 |

All fixed-budget endpoints are retained; the last mini-batch loss is not a validation score.
Losses, gradients, learning rate and heartbeat are retained locally in hash-bound fit receipts.
The legacy control adds 4,000 updates; total new execution is 40,000 updates, not 36,000 plus a separate pilot budget.
