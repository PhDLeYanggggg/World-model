# Coupled Risk Evidence Gate

## Material Passport

Registered fitting screen; NOT the M3W deployment or submission gate.
Evidence:registration.json,72 bound group receipts,summary.json,verification.json.

| Prespecified condition | Result |
|---|---|
| All views retain full local reference support | pass |
| Primary easy MSE beats original | FAIL:0.000326962 >0.000309373 |
| Primary easy MSE beats independent cost | pass |
| Primary easy MSE beats uniform selected neighbor | pass |
| Primary all MSE no worse than independent cost | pass |
| Primary all MSE no worse than uniform selected neighbor | pass |
| At least12/18 pairs beat original and independent easy MSE | FAIL:1/18 |
| Joint moment constraints hold | pass |

Overall:FAIL,6/8. Secondary conditional assignment:FAIL,1/2 (beats shuffled,
loses mean-alpha). No gate is relaxed and no secondary arm replaces the primary.

Finite local support, numerical feasibility and lower donor training loss are
not calibrated risk guarantees. The2% positive selected harm/reference budget
is unchanged, not reinterpreted as2% signed-risk MSE or easy ADE degradation.

Whole-pipeline OOF:false. Independent confirmation:not_run. Neural dynamics
training:not_run. New deployment:false. Stage5C:false. SMC:false.
