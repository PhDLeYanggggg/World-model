# Fitting Magnitude and Target-Definition Drift

Descriptive only. No slope,cut,model or deployment policy was selected from these summaries.

| Family/arm | H_all slope min/median/max | H_easy slope min/median/max | Bound components low/high |
|---|---|---|---|
| full/cost_only | [0.01088069453210158, 0.3795181745363312, 0.8290719874723838] | [0.0035872887100152594, 0.32669878692089294, 1.190909047761497] | 0/0 |
| full/cap_aux | [0.014373657510553563, 0.3815808681216727, 0.8838180278200594] | [0.0035361684849386193, 0.3184098620156035, 1.2607763104744443] | 0/0 |
| full/shuffled_aux | [0.010470278829220457, 0.3631669169710853, 0.9440188888180605] | [0.002479654614724553, 0.31538763612597365, 1.2635645296179077] | 0/0 |
| motion_only/cost_only | [0.014478817731579123, 0.42735572873198496, 0.8779409917263585] | [0.0003659871434529169, 0.05380202853459748, 1.345865798391725] | 0/0 |
| motion_only/cap_aux | [0.017570409768109092, 0.4322542678268221, 1.0373793758062604] | [0.00084141326219017, 0.06018149290693328, 1.4791615141082883] | 0/0 |
| motion_only/shuffled_aux | [0.01593877779660168, 0.4238870089141591, 0.8789379966897313] | [0.0004718844913982957, 0.05189607439868803, 1.4353437138632084] | 0/0 |

full inner event support: `{"prior": {"count": 216, "minimum": 0.02037318410910169, "median": 0.0472702094554181, "maximum": 0.10322179388201297}, "zero_positive_fits": 0, "positive_rows": {"count": 216, "minimum": 67.0, "median": 240.5, "maximum": 3794.0}}`


motion_only inner event support: `{"prior": {"count": 216, "minimum": 0.00195944671030669, "median": 0.011958653079568074, "maximum": 0.0468854087002743}, "zero_positive_fits": 0, "positive_rows": {"count": 216, "minimum": 5.0, "median": 39.0, "maximum": 1732.0}}`

The JSON retains every outer-view cut drift. The readout is trained against producer-specific
two-locality easy definitions and applied to three-locality outer heads. Single-locality cap-reference
training is a further support/transport limitation. None of these diagnostics proves causation.
Obs8/pred12 native steps,detector pixels,source development only. Stage5C/SMC off.
