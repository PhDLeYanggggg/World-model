# Event Support and Residual Ceiling

## Scope
Fresh diagnostic computations over frozen inner fitting-locality predictions.
No refitting, threshold change or favorable subgroup selection. Counts are dependent views.

| Family | Views | <10 event tracks | <10 harm-mass effective tracks | <=1 event recording | Cap floor / raw MSE, min / median / max |
|---|---:|---:|---:|---:|---|
| full | 216 | 3 | 35 | 0 | {'n': 216, 'min': 0.016018536469143273, 'median': 0.651767140553343, 'max': 0.9991654682610057} |
| motion_only | 216 | 66 | 158 | 15 | {'n': 216, 'min': 0.0, 'median': 0.7075232215305796, 'max': 0.9996879517431956} |

The cap floor is label-derived mean(max(true easy harm - frozen predicted all harm,0)^2).
It is a lower bound for this easy-only clipped probe, not all possible risk models or dynamics models.
Harm-mass effective count describes concentration, not an independent sample-size or power calculation.

## Absolute Cost Companion

Native pixel-cost squared MSE across dependent views, not pooled trajectory ADE/FDE or an independent sample.
A percentage contrast can be large when its control MSE is tiny; these values retain that denominator context.

| Family / arm | Easy-harm MSE, min / median / max | Coverage, min / median / max |
|---|---|---|
| full / raw | {'n': 216, 'min': 0.001475876843991543, 'median': 0.04754350088505328, 'max': 2.0560532927071415} | {'n': 216, 'min': 2.9700572861438154e-05, 'median': 0.6493198082423868, 'max': 6.058325329723427} |
| full / score_only | {'n': 216, 'min': 0.001509958866276074, 'median': 0.046862884782281906, 'max': 2.053056322969861} | {'n': 216, 'min': 0.00825428537857388, 'median': 0.9936209961501921, 'max': 30.158960960482727} |
| full / old_summary | {'n': 216, 'min': 0.0014826966282298164, 'median': 0.045774071137788384, 'max': 2.048105664962825} | {'n': 216, 'min': 0.0077254095959677855, 'median': 0.9929204028318857, 'max': 29.02099418544323} |
| full / ordered_history | {'n': 216, 'min': 0.0015058555850509438, 'median': 0.04646425868515292, 'max': 2.0521010388046994} | {'n': 216, 'min': 0.00842125556438757, 'median': 1.0938634843034336, 'max': 28.664543898837138} |
| full / history_neighbors | {'n': 216, 'min': 0.001514238115529214, 'median': 0.04605273597030712, 'max': 2.051874809585315} | {'n': 216, 'min': 0.007966522154774839, 'median': 1.0595088277474458, 'max': 28.099676894120133} |
| motion_only / raw | {'n': 216, 'min': 7.404402062432485e-07, 'median': 0.02630461027581698, 'max': 2.13074246950153} | {'n': 213, 'min': 0.00022482017116356494, 'median': 0.4466655680349615, 'max': 1061.5161024488064} |
| motion_only / score_only | {'n': 216, 'min': 9.030415776003912e-07, 'median': 0.026641168152210397, 'max': 2.1282178769499946} | {'n': 213, 'min': 0.006976612960292682, 'median': 2.0346662836518723, 'max': 4262.871027098311} |
| motion_only / old_summary | {'n': 216, 'min': 6.871030736208608e-07, 'median': 0.026630250812429458, 'max': 2.1239261643853564} | {'n': 213, 'min': 0.007819060779035264, 'median': 1.658912511651391, 'max': 1633.0642892227554} |
| motion_only / ordered_history | {'n': 216, 'min': 8.989521026427418e-06, 'median': 0.0265979493698836, 'max': 2.124297322839834} | {'n': 213, 'min': 0.007627886181658583, 'median': 2.3330199830662224, 'max': 7731.2382811203115} |
| motion_only / history_neighbors | {'n': 216, 'min': 1.2199630110056581e-05, 'median': 0.026640732211865477, 'max': 2.125211174974706} | {'n': 213, 'min': 0.007548693604608613, 'median': 1.9228642996820293, 'max': 6665.759683815266} |

## Fitting to Inner Transfer

These row-weighted within-view summaries are descriptive. Training optimizes equal-locality loss,
so a pooled fitting MSE change is not itself an optimization certificate.

| Family / control | Better on both | Fitting only | Inner only | Neither |
|---|---:|---:|---:|---:|
| full / score_only | 101 | 77 | 26 | 12 |
| full / old_summary | 68 | 82 | 18 | 48 |
| full / ordered_history | 68 | 106 | 26 | 16 |
| motion_only / score_only | 108 | 58 | 29 | 21 |
| motion_only / old_summary | 38 | 124 | 9 | 45 |
| motion_only / ordered_history | 38 | 120 | 27 | 31 |

## Locality Support

Min/median/max include seeds, overlapping producer assignments, changing fitting cuts and outer contexts.
A support count is never summed across these replicas.

| Family / locality | Event tracks | Effective event tracks | Largest track share |
|---|---|---|---|
| full / eu-locality-007 | {'n': 18, 'min': 91.0, 'median': 106.5, 'max': 120.0} | {'n': 18, 'min': 8.699846708989284, 'median': 28.20034742193244, 'max': 43.42455995838004} | {'n': 18, 'min': 0.06071896323674694, 'median': 0.10002183340683461, 'max': 0.29773203299342166} |
| full / eu-locality-008 | {'n': 18, 'min': 1971.0, 'median': 2548.0, 'max': 3428.0} | {'n': 18, 'min': 263.9819614360558, 'median': 408.34457910260755, 'max': 709.5950241444276} | {'n': 18, 'min': 0.007534943789524463, 'median': 0.012810969180678458, 'max': 0.02624576535355317} |
| full / eu-locality-020 | {'n': 18, 'min': 14.0, 'median': 22.0, 'max': 27.0} | {'n': 18, 'min': 3.3168772550643624, 'median': 5.939570916501651, 'max': 10.546835705416143} | {'n': 18, 'min': 0.16080079128919805, 'median': 0.26739178023089877, 'max': 0.43564883324305714} |
| full / eu-locality-048 | {'n': 18, 'min': 79.0, 'median': 94.0, 'max': 124.0} | {'n': 18, 'min': 16.9687179085086, 'median': 20.835586353684224, 'max': 31.70292288646364} | {'n': 18, 'min': 0.07619887258481173, 'median': 0.1339512764976803, 'max': 0.17112585341385403} |
| full / eu-locality-067 | {'n': 18, 'min': 86.0, 'median': 102.0, 'max': 124.0} | {'n': 18, 'min': 16.60462488327212, 'median': 24.294396041645435, 'max': 34.284617783730816} | {'n': 18, 'min': 0.07915680818597239, 'median': 0.1287483384077785, 'max': 0.1811301389773084} |
| full / eu-locality-074 | {'n': 18, 'min': 582.0, 'median': 1677.0, 'max': 2857.0} | {'n': 18, 'min': 133.6540905834102, 'median': 499.4826899060819, 'max': 939.0641300583223} | {'n': 18, 'min': 0.005068598730797084, 'median': 0.009021121883974281, 'max': 0.02556229833660128} |
| full / eu-locality-082 | {'n': 18, 'min': 15.0, 'median': 26.5, 'max': 34.0} | {'n': 18, 'min': 4.230140299222762, 'median': 8.142515844940595, 'max': 13.131815857967947} | {'n': 18, 'min': 0.12414146354738317, 'median': 0.2666386597625494, 'max': 0.43895325054814616} |
| full / eu-locality-110 | {'n': 18, 'min': 41.0, 'median': 90.5, 'max': 111.0} | {'n': 18, 'min': 18.66234097103656, 'median': 30.208434261310487, 'max': 35.570622952433084} | {'n': 18, 'min': 0.07258416784817896, 'median': 0.0897386684074993, 'max': 0.11721866794703488} |
| full / eu-locality-112 | {'n': 18, 'min': 8.0, 'median': 44.5, 'max': 80.0} | {'n': 18, 'min': 3.5127981163597983, 'median': 13.495491166632252, 'max': 29.942044776231448} | {'n': 18, 'min': 0.06265617133152662, 'median': 0.1560388923509489, 'max': 0.47204900313504744} |
| full / eu-locality-119 | {'n': 18, 'min': 73.0, 'median': 92.0, 'max': 116.0} | {'n': 18, 'min': 18.471174634584802, 'median': 30.304680529223383, 'max': 39.79967574248814} | {'n': 18, 'min': 0.058901478190747786, 'median': 0.08312028269228311, 'max': 0.11256539251741701} |
| full / eu-locality-124 | {'n': 18, 'min': 24.0, 'median': 35.5, 'max': 43.0} | {'n': 18, 'min': 11.707436388335902, 'median': 17.94985391038772, 'max': 24.305242496727402} | {'n': 18, 'min': 0.07025475966231262, 'median': 0.10782597087361456, 'max': 0.191950955929626} |
| full / eu-locality-126 | {'n': 18, 'min': 124.0, 'median': 162.5, 'max': 186.0} | {'n': 18, 'min': 22.837368752833736, 'median': 42.606348450316425, 'max': 68.60851080926754} | {'n': 18, 'min': 0.03626879891723591, 'median': 0.06497708686311021, 'max': 0.16201986961996237} |
| motion_only / eu-locality-007 | {'n': 18, 'min': 16.0, 'median': 20.5, 'max': 24.0} | {'n': 18, 'min': 4.687828754095964, 'median': 6.423447448511316, 'max': 8.24905600543411} | {'n': 18, 'min': 0.18812689854570266, 'median': 0.23642453988450393, 'max': 0.3104752498511818} |
| motion_only / eu-locality-008 | {'n': 18, 'min': 356.0, 'median': 546.5, 'max': 987.0} | {'n': 18, 'min': 93.99921273888411, 'median': 148.53308035063003, 'max': 310.234982444255} | {'n': 18, 'min': 0.009117205395593665, 'median': 0.021271075860054965, 'max': 0.03489970600359421} |
| motion_only / eu-locality-020 | {'n': 18, 'min': 5.0, 'median': 9.0, 'max': 15.0} | {'n': 18, 'min': 2.519220961547911, 'median': 4.382330219744018, 'max': 8.463641397433081} | {'n': 18, 'min': 0.1696896948824961, 'median': 0.35721412101657046, 'max': 0.58320935417592} |
| motion_only / eu-locality-048 | {'n': 18, 'min': 23.0, 'median': 27.5, 'max': 41.0} | {'n': 18, 'min': 3.0910246668767756, 'median': 7.101371291710979, 'max': 15.807194733655567} | {'n': 18, 'min': 0.14766703564729472, 'median': 0.3348962521703714, 'max': 0.5469950167502472} |
| motion_only / eu-locality-067 | {'n': 18, 'min': 14.0, 'median': 19.5, 'max': 23.0} | {'n': 18, 'min': 5.589759187912075, 'median': 6.588254821022291, 'max': 7.5955288107594505} | {'n': 18, 'min': 0.21922648922028823, 'median': 0.28999462681268556, 'max': 0.32204056242206797} |
| motion_only / eu-locality-074 | {'n': 18, 'min': 212.0, 'median': 783.0, 'max': 1477.0} | {'n': 18, 'min': 74.24761467509879, 'median': 308.5107776521638, 'max': 598.5026826791747} | {'n': 18, 'min': 0.00837520245822546, 'median': 0.016688549134769165, 'max': 0.042873579950771304} |
| motion_only / eu-locality-082 | {'n': 18, 'min': 2.0, 'median': 4.5, 'max': 10.0} | {'n': 18, 'min': 1.0910485348414982, 'median': 2.3587218539379817, 'max': 5.386501244393466} | {'n': 18, 'min': 0.26892469413949255, 'median': 0.6153690809345974, 'max': 0.956371290729602} |
| motion_only / eu-locality-110 | {'n': 18, 'min': 1.0, 'median': 6.0, 'max': 15.0} | {'n': 18, 'min': 1.0, 'median': 2.3526013173460045, 'max': 7.090611975987571} | {'n': 18, 'min': 0.24601945228084915, 'median': 0.6078835716646254, 'max': 1.0} |
| motion_only / eu-locality-112 | {'n': 18, 'min': 0.0, 'median': 12.0, 'max': 29.0} | {'n': 18, 'min': 0.0, 'median': 3.391978433565898, 'max': 5.764277263520659} | {'n': 15, 'min': 0.2706496908080729, 'median': 0.4562915812973777, 'max': 1.0} |
| motion_only / eu-locality-119 | {'n': 18, 'min': 23.0, 'median': 38.5, 'max': 62.0} | {'n': 18, 'min': 8.89233015676866, 'median': 12.781039965493456, 'max': 21.882207505075442} | {'n': 18, 'min': 0.08458940948151956, 'median': 0.1423382352597879, 'max': 0.21501491332922718} |
| motion_only / eu-locality-124 | {'n': 18, 'min': 2.0, 'median': 5.0, 'max': 8.0} | {'n': 18, 'min': 1.3901546785675154, 'median': 2.4998051752441977, 'max': 3.4928033926246886} | {'n': 18, 'min': 0.46122495313167194, 'median': 0.5815472974707362, 'max': 0.831167942374254} |
| motion_only / eu-locality-126 | {'n': 18, 'min': 26.0, 'median': 31.0, 'max': 36.0} | {'n': 18, 'min': 5.454795072863922, 'median': 7.944096360861598, 'max': 10.909461739382765} | {'n': 18, 'min': 0.24024353765692893, 'median': 0.26215570377174896, 'max': 0.36101941884676664} |

Source-development only; obs8/pred12 sampled annotation steps, detector pixels.
No seconds/metric/physical-safety/true3D/foundation claim. Stage5C/SMC remain off.
