# User Action Required: Stage42-FY Horizon Retry Decision

Do not run more same-feature h100 weak-horizon model retries until these blockers are closed.

| weak key | required before retry | next action ids |
| --- | --- | --- |
| `TrajNet|100` | official longer TrajNet-compatible raw source<br>terms confirmation<br>guarded conversion<br>no-leakage audit<br>train-only source-CV | FW-H100-TrajNet|100<br>FW-DOMAIN-TrajNet<br>FW-TERMS-trajnetplusplus_official |
| `UCY|100` | UCY original terms/user confirmation<br>guarded conversion of local h100 candidates<br>no-leakage audit<br>train-only source-CV | FW-TERMS-ucy_crowd_original<br>FW-H100-UCY|100<br>FW-DOMAIN-UCY |
