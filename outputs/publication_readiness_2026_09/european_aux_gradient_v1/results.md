# Fitting-Only Gradient and AdamW Results

fresh_run: 17,280 isolated virtual AdamW updates from 432 frozen step-2000 states;
3,456 final-state batch diagnostics and 144 initial checks. No new fully trained model.
cached_verified: fitting features, nested targets, preprocessing, checkpoints and moments.
not_run: intermediate trajectory checkpoints, new full training, held readout, new policy,
independent selection/calibration/confirmation. No deployment change.

Positive percentage means lower actual fitting-probe loss after the isolated step.
Not trajectory improvement, held-out performance, physical safety or a causal account of training failure.
Probe rows are disjoint from the single update, but all are previously trained on.

## Primary Repair Screen
```json
{
  "projection_minus_true/cost4": false,
  "projection_minus_true/easy_harm_positive": false,
  "projected_true_minus_shuffled/cost4": false,
  "projected_true_minus_shuffled/easy_harm_positive": false
}
```
Separate projection training warranted: False

## Full-Input Cap-Auxiliary Frozen States

| Assignment | Contrast | Metric | Point (%) | 95% descriptive locality interval |
|---|---|---|---:|---|
| P0__C1 | projected_true_minus_shuffled | cost4 | 0.000722798616595516 | [-0.0008537737871359973, 0.003597082004425364] |
| P0__C1 | projected_true_minus_shuffled | easy_harm_all | 0.00042045893201969516 | [9.321789655902925e-06, 0.0010770755756791585] |
| P0__C1 | projected_true_minus_shuffled | easy_harm_positive | 0.00042135355713022934 | [5.1158418414435e-05, 0.0011436963115166229] |
| P0__C1 | projection_minus_true | cost4 | 2.486373695088482e-06 | [-1.354399255906228e-05, 2.567839991089365e-05] |
| P0__C1 | projection_minus_true | easy_harm_all | 2.8397880618824356e-06 | [-2.0199187806733976e-07, 1.777514101920696e-05] |
| P0__C1 | projection_minus_true | easy_harm_positive | 3.1428526501807267e-06 | [1.0069795236264817e-07, 1.952109926717027e-05] |
| P0__C1 | true_minus_cost | cost4 | 0.0003287552494850763 | [-0.00021788564444429626, 0.0013253769249302186] |
| P0__C1 | true_minus_cost | easy_harm_all | -0.00012417815103222914 | [-0.00016501227189161217, 4.953833230243235e-05] |
| P0__C1 | true_minus_cost | easy_harm_positive | -0.0001248694254558894 | [-0.0001634197071476579, 4.667259355233112e-05] |
| P0__C1 | true_minus_shuffled | cost4 | 0.0007113354765429972 | [-0.0008625917013907979, 0.0035807959701693363] |
| P0__C1 | true_minus_shuffled | easy_harm_all | 0.0004104611987670636 | [-5.235383352066876e-06, 0.001047670794334571] |
| P0__C1 | true_minus_shuffled | easy_harm_positive | 0.00040961045432910074 | [3.6399983231935264e-05, 0.001108848766157202] |
| P0__C2 | projected_true_minus_shuffled | cost4 | -0.002417788923693089 | [-0.006752234582187701, 0.0007905293110803221] |
| P0__C2 | projected_true_minus_shuffled | easy_harm_all | -0.011689054305584387 | [-0.02106048489919637, -0.00022685898673903974] |
| P0__C2 | projected_true_minus_shuffled | easy_harm_positive | -0.018354138194935206 | [-0.0222543479880456, -0.00019365555352869086] |
| P0__C2 | projection_minus_true | cost4 | -1.2147587089704327e-08 | [-9.94372536982211e-05, 5.176002413033377e-05] |
| P0__C2 | projection_minus_true | easy_harm_all | -0.00012306526602946845 | [-0.0002554147159989468, 8.236325357079439e-05] |
| P0__C2 | projection_minus_true | easy_harm_positive | -0.00024567589630740503 | [-0.0003075232556748568, 8.727668815003828e-05] |
| P0__C2 | true_minus_cost | cost4 | -0.0008073516875892773 | [-0.0026571578591667016, 0.000257808971301852] |
| P0__C2 | true_minus_cost | easy_harm_all | -0.0043647981724392534 | [-0.007872022910302909, -5.380880308284331e-05] |
| P0__C2 | true_minus_cost | easy_harm_positive | -0.006866795012300459 | [-0.00833393485903661, -0.0001134589611765158] |
| P0__C2 | true_minus_shuffled | cost4 | -0.002194202195759494 | [-0.007204177807136934, 0.001364525379237855] |
| P0__C2 | true_minus_shuffled | easy_harm_all | -0.011996749612135343 | [-0.022033005151484975, 0.00023548534931951717] |
| P0__C2 | true_minus_shuffled | easy_harm_positive | -0.019156564653165228 | [-0.023334763728900684, 2.7313840046298443e-05] |
| P1__C0 | projected_true_minus_shuffled | cost4 | -0.001225315790427045 | [-0.0024275422668091325, 0.001262982785452948] |
| P1__C0 | projected_true_minus_shuffled | easy_harm_all | -0.0032779835082303215 | [-0.005904287868945032, -0.0012305715166894252] |
| P1__C0 | projected_true_minus_shuffled | easy_harm_positive | -0.0031834120318137956 | [-0.005642611879747672, -0.0012816478296755931] |
| P1__C0 | projection_minus_true | cost4 | 2.5992630372637953e-05 | [-3.905048903424768e-05, 8.365056637975e-05] |
| P1__C0 | projection_minus_true | easy_harm_all | 6.206674871862606e-05 | [-5.983591298204608e-05, 0.00031454582461431407] |
| P1__C0 | projection_minus_true | easy_harm_positive | 7.106959941699379e-05 | [-5.9763015146551226e-05, 0.0003569206992601565] |
| P1__C0 | true_minus_cost | cost4 | 0.00064751574036465 | [-4.606764669596522e-08, 0.0011072161596456735] |
| P1__C0 | true_minus_cost | easy_harm_all | -0.00036024108059678643 | [-0.003934078454094612, 0.0002952571647016807] |
| P1__C0 | true_minus_cost | easy_harm_positive | -0.0004420493982411205 | [-0.003749757035949707, 0.00032639830623678467] |
| P1__C0 | true_minus_shuffled | cost4 | -0.0012718979176039372 | [-0.0025756836565059412, 0.0014265914045114914] |
| P1__C0 | true_minus_shuffled | easy_harm_all | -0.00340832082913102 | [-0.00647969842710323, -0.0012110807275155632] |
| P1__C0 | true_minus_shuffled | easy_harm_positive | -0.0033205002972457867 | [-0.006228266119984049, -0.0013127812527530464] |
| P1__C2 | projected_true_minus_shuffled | cost4 | -0.0016533355286699662 | [-0.0029126921428131733, -0.0003668441384397397] |
| P1__C2 | projected_true_minus_shuffled | easy_harm_all | -0.0008546632598813944 | [-0.006783080927261589, 0.0011423701153924811] |
| P1__C2 | projected_true_minus_shuffled | easy_harm_positive | -0.0017113942758298591 | [-0.005285170271488167, 0.0012717307756833134] |
| P1__C2 | projection_minus_true | cost4 | 2.6080077365316357e-05 | [-1.760796491409359e-05, 5.7181627452370416e-05] |
| P1__C2 | projection_minus_true | easy_harm_all | -1.739797304035843e-05 | [-0.000103761027228329, 3.01702264305234e-05] |
| P1__C2 | projection_minus_true | easy_harm_positive | -6.0647055319830175e-05 | [-0.00014851816797599834, 2.812816227106951e-05] |
| P1__C2 | true_minus_cost | cost4 | -0.00022184726890425745 | [-0.0006597244974608956, 0.00025952601570899107] |
| P1__C2 | true_minus_cost | easy_harm_all | -0.0007939423996737235 | [-0.0036159453656175325, 0.0007823631707041474] |
| P1__C2 | true_minus_cost | easy_harm_positive | -0.001335350114235434 | [-0.0033607936670044364, 0.0007872098030933375] |
| P1__C2 | true_minus_shuffled | cost4 | -0.0016652467267029636 | [-0.003318863232625834, -0.00028332268211790363] |
| P1__C2 | true_minus_shuffled | easy_harm_all | -0.001082625026113733 | [-0.008011091970268334, 0.0012251111555191161] |
| P1__C2 | true_minus_shuffled | easy_harm_positive | -0.0021882791754642577 | [-0.006463298593958695, 0.0013801409882866831] |
| P2__C0 | projected_true_minus_shuffled | cost4 | -0.0028529188195839487 | [-0.008810791369892506, 0.0024559298084756527] |
| P2__C0 | projected_true_minus_shuffled | easy_harm_all | -1.8841569358827034e-05 | [-0.002884452305498971, 0.0020989847147883053] |
| P2__C0 | projected_true_minus_shuffled | easy_harm_positive | 0.00020416025319584813 | [-0.0014276985880755269, 0.002211482814988622] |
| P2__C0 | projection_minus_true | cost4 | 5.485302499450994e-06 | [-3.8664670756219325e-06, 1.044822397718115e-05] |
| P2__C0 | projection_minus_true | easy_harm_all | -2.760808406404241e-06 | [-1.6563078160098494e-05, 1.2734253523735445e-06] |
| P2__C0 | projection_minus_true | easy_harm_positive | -3.423492657660524e-06 | [-1.7038481480003313e-05, 1.0856364130110041e-06] |
| P2__C0 | true_minus_cost | cost4 | -0.00125142598917215 | [-0.0027010866520673776, -0.00043672929887565904] |
| P2__C0 | true_minus_cost | easy_harm_all | -0.002088792754762784 | [-0.007947513377602394, -0.0006439595335719782] |
| P2__C0 | true_minus_cost | easy_harm_positive | -0.002420807746832635 | [-0.006807770882016114, -0.0007259132481022969] |
| P2__C0 | true_minus_shuffled | cost4 | -0.0033776799434480057 | [-0.010751915172064605, 0.002445874274889608] |
| P2__C0 | true_minus_shuffled | easy_harm_all | 2.6941593383990807e-05 | [-0.003944098510696594, 0.0025283063507733528] |
| P2__C0 | true_minus_shuffled | easy_harm_positive | 0.0002165216389084871 | [-0.002242174255995037, 0.00261713597196222] |
| P2__C1 | projected_true_minus_shuffled | cost4 | 0.00010587832630986587 | [-0.002455829984028306, 0.002630664516970794] |
| P2__C1 | projected_true_minus_shuffled | easy_harm_all | 0.003739170746256229 | [-0.0010848415509808922, 0.01303819139594187] |
| P2__C1 | projected_true_minus_shuffled | easy_harm_positive | 0.0035849298068091574 | [-0.0002129990840215958, 0.012009777157753462] |
| P2__C1 | projection_minus_true | cost4 | -4.13740156156271e-06 | [-1.0056777121826994e-05, 1.8682319175166573e-06] |
| P2__C1 | projection_minus_true | easy_harm_all | -1.8288743000443406e-05 | [-2.6023560796856504e-05, -1.194865727436619e-05] |
| P2__C1 | projection_minus_true | easy_harm_positive | -2.105048934370796e-05 | [-2.7173297716650397e-05, -1.4005686542802532e-05] |
| P2__C1 | true_minus_cost | cost4 | -1.7796971329192928e-05 | [-0.0006258642279290226, 0.0004937055034524038] |
| P2__C1 | true_minus_cost | easy_harm_all | 0.000715720767678595 | [-0.0012762487278062846, 0.0041195682664339015] |
| P2__C1 | true_minus_cost | easy_harm_positive | 0.0010864866933364694 | [-0.0009905425085102201, 0.004537421407162854] |
| P2__C1 | true_minus_shuffled | cost4 | 0.0009274310508662444 | [-0.0019257857500876126, 0.0037395082531501136] |
| P2__C1 | true_minus_shuffled | easy_harm_all | 0.00488157779590613 | [-0.0003435248430379836, 0.015658731578551535] |
| P2__C1 | true_minus_shuffled | easy_harm_positive | 0.004640136435624866 | [0.0006033524344824917, 0.01427840361513702] |

All assignments, full/motion families, three frozen arms and gradients are in aggregate_metrics.json.
Locality averages combine eight repeats, three fitting contexts and three seeds; four localities are
resampled 3000 times. Models and assignments share fitting data. These intervals are descriptive,
not independent confirmation or a generalization guarantee. No window-level inferential sample size.
Projection protects the raw shared four-cost gradient only. AdamW moments, clipping and finite steps
can change actual effects, and easy-harm is not individually protected. Null gradients are unsupported.
End-state diagnostics cannot establish early/mid-training behavior. Known zero-envelope costs remain included.
Obs8/pred12 annotation steps, detector pixels. No metric/seconds, physical-safety, human-gold,
true3D/foundation claims, Stage5C execution or SMC.
