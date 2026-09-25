# Frozen Six-Locality Readout

Fresh inference/readout of frozen, hash-verified checkpoints; no new training.
Model-selection evidence, not final confirmation. All 36 dependent views share the same six localities.

| Policy | All ADE gain vs old incumbent | Hard gain | Worst-locality easy degradation range | Easy-pass views | Positive/negative CIs |
|---|---:|---:|---:|---:|---:|
| floor_reference | -0.120060% to +1.363791% | -1.144424% to +2.607474% | +0.000000% to +7.497147% | 29/36 | 16/0 |
| incumbent_reference | +0.008471% to +0.743860% | -0.006870% to +1.331650% | +0.000000% to +8.327380% | 29/36 | 30/0 |
| add_only | +0.010488% to +0.778997% | -0.004323% to +1.355692% | +0.000000% to +8.216907% | 29/36 | 33/0 |
| remove_only | -0.037340% to +0.005190% | -0.024042% to +0.011268% | +0.000000% to +6.300844% | 30/36 | 0/27 |
| ridge_incumbent | -1.198595% to +4.061104% | -0.468478% to +5.709516% | +0.000000% to +6.024246% | 32/36 | 17/7 |
| old_stop | +0.000000% to +0.000000% | +0.000000% to +0.000000% | +0.000000% to +6.076432% | 30/36 | 0/0 |
| previous_matched | -0.100147% to +1.402259% | -0.847555% to +2.549735% | +0.000000% to +8.327429% | 28/36 | 17/0 |
| raw_neural | -16.065348% to +12.907663% | -1.390264% to +18.647633% | +24.774235% to +30.675096% | 0/36 | 12/24 |

A positive average is not an easy-risk certificate. Exact-zero CV harm is assessed separately.
CIs are paired locality bootstrap (3,000 draws), exploratory and not multiplicity-adjusted.
The table reports ranges over all source/controller/event/seed views, not independent replications.
No favorable seed or candidate is selected. Calibration and confirmation remain closed.

## Three-Seed Means

These estimates average seedwise relative gains within each locality before bootstrapping six localities.
They are not the performance of an averaged-trajectory ensemble.

| Policy | Producer/controller/event | All gain | 95% locality CI | Hard gain |
|---|---|---:|---:|---:|
| floor_reference | producer0_controller1_all | +1.281835% | +0.732309% to +1.855575% | +2.516215% |
| floor_reference | producer0_controller1_easy | +0.173049% | -0.036153% to +0.389552% | -0.010760% |
| floor_reference | producer0_controller2_all | +0.271914% | -0.157339% to +0.857456% | +0.598878% |
| floor_reference | producer0_controller2_easy | -0.039405% | -0.126269% to +0.016399% | -0.009719% |
| floor_reference | producer1_controller0_all | +0.595856% | +0.172877% to +1.231528% | -0.462588% |
| floor_reference | producer1_controller0_easy | +0.042294% | -0.011813% to +0.099735% | -0.044003% |
| floor_reference | producer1_controller2_all | +0.347570% | -0.211307% to +1.302691% | -0.340622% |
| floor_reference | producer1_controller2_easy | +0.080699% | -0.032495% to +0.220956% | +0.011140% |
| floor_reference | producer2_controller0_all | +0.234893% | +0.064608% to +0.471725% | +0.297823% |
| floor_reference | producer2_controller0_easy | -0.005700% | -0.082060% to +0.064789% | -0.006435% |
| floor_reference | producer2_controller1_all | +0.490806% | +0.139299% to +0.944995% | +0.683844% |
| floor_reference | producer2_controller1_easy | +0.122483% | +0.051049% to +0.202740% | +0.132678% |
| incumbent_reference | producer0_controller1_all | +0.705226% | +0.373705% to +1.075597% | +1.180968% |
| incumbent_reference | producer0_controller1_easy | +0.274205% | +0.114966% to +0.481974% | +0.010231% |
| incumbent_reference | producer0_controller2_all | +0.311830% | +0.124564% to +0.501719% | +0.463246% |
| incumbent_reference | producer0_controller2_easy | +0.095037% | +0.025984% to +0.189427% | -0.000224% |
| incumbent_reference | producer1_controller0_all | +0.407196% | +0.175002% to +0.770773% | +0.141467% |
| incumbent_reference | producer1_controller0_easy | +0.098206% | +0.031025% to +0.172039% | +0.001346% |
| incumbent_reference | producer1_controller2_all | +0.135341% | +0.054242% to +0.235452% | +0.134486% |
| incumbent_reference | producer1_controller2_easy | +0.084821% | +0.040791% to +0.133625% | +0.000325% |
| incumbent_reference | producer2_controller0_all | +0.098491% | +0.022434% to +0.176351% | +0.066081% |
| incumbent_reference | producer2_controller0_easy | +0.029072% | -0.000697% to +0.081748% | +0.023381% |
| incumbent_reference | producer2_controller1_all | +0.122846% | +0.041559% to +0.212056% | +0.101035% |
| incumbent_reference | producer2_controller1_easy | +0.052292% | +0.002212% to +0.102893% | +0.059940% |
| add_only | producer0_controller1_all | +0.737852% | +0.401408% to +1.117218% | +1.188830% |
| add_only | producer0_controller1_easy | +0.279392% | +0.120193% to +0.489383% | +0.016539% |
| add_only | producer0_controller2_all | +0.327496% | +0.139087% to +0.521382% | +0.466628% |
| add_only | producer0_controller2_easy | +0.100609% | +0.031353% to +0.197489% | +0.002015% |
| add_only | producer1_controller0_all | +0.420290% | +0.185109% to +0.790397% | +0.141212% |
| add_only | producer1_controller0_easy | +0.100377% | +0.032854% to +0.174065% | +0.001451% |
| add_only | producer1_controller2_all | +0.142853% | +0.059507% to +0.242093% | +0.135472% |
| add_only | producer1_controller2_easy | +0.091688% | +0.043718% to +0.142576% | +0.001918% |
| add_only | producer2_controller0_all | +0.105834% | +0.027584% to +0.189378% | +0.073676% |
| add_only | producer2_controller0_easy | +0.028718% | -0.003414% to +0.083371% | +0.020884% |
| add_only | producer2_controller1_all | +0.125179% | +0.043179% to +0.211938% | +0.112512% |
| add_only | producer2_controller1_easy | +0.055959% | +0.002604% to +0.112050% | +0.062453% |
| remove_only | producer0_controller1_all | -0.032626% | -0.051427% to -0.014384% | -0.007861% |
| remove_only | producer0_controller1_easy | -0.005187% | -0.011263% to -0.001377% | -0.006309% |
| remove_only | producer0_controller2_all | -0.015667% | -0.026059% to -0.006816% | -0.003381% |
| remove_only | producer0_controller2_easy | -0.005572% | -0.009526% to -0.002083% | -0.002239% |
| remove_only | producer1_controller0_all | -0.013094% | -0.019329% to -0.006739% | +0.000255% |
| remove_only | producer1_controller0_easy | -0.002171% | -0.004640% to -0.000486% | -0.000105% |
| remove_only | producer1_controller2_all | -0.007512% | -0.012849% to -0.002179% | -0.000986% |
| remove_only | producer1_controller2_easy | -0.006867% | -0.011427% to -0.002308% | -0.001593% |
| remove_only | producer2_controller0_all | -0.007343% | -0.021240% to +0.001285% | -0.007595% |
| remove_only | producer2_controller0_easy | +0.000353% | -0.001599% to +0.002671% | +0.002497% |
| remove_only | producer2_controller1_all | -0.002332% | -0.005752% to +0.000590% | -0.011477% |
| remove_only | producer2_controller1_easy | -0.003667% | -0.010175% to -0.000003% | -0.002512% |
| ridge_incumbent | producer0_controller1_all | -0.072431% | -0.249072% to +0.068236% | +0.117460% |
| ridge_incumbent | producer0_controller1_easy | -0.186354% | -0.513574% to +0.260202% | +0.188439% |
| ridge_incumbent | producer0_controller2_all | -0.091739% | -0.205516% to +0.045407% | +0.140269% |
| ridge_incumbent | producer0_controller2_easy | +0.920007% | +0.391289% to +1.574969% | +1.177512% |
| ridge_incumbent | producer1_controller0_all | -0.296310% | -0.522301% to -0.041592% | +0.287940% |
| ridge_incumbent | producer1_controller0_easy | -0.940466% | -1.976020% to -0.083023% | +0.357531% |
| ridge_incumbent | producer1_controller2_all | -0.315259% | -0.641128% to -0.061985% | -0.317996% |
| ridge_incumbent | producer1_controller2_easy | +1.156574% | +0.552396% to +1.843026% | +1.892556% |
| ridge_incumbent | producer2_controller0_all | +0.131702% | +0.057973% to +0.202870% | -0.019200% |
| ridge_incumbent | producer2_controller0_easy | +2.199469% | +1.219514% to +3.307931% | +3.019139% |
| ridge_incumbent | producer2_controller1_all | +0.629714% | +0.282144% to +1.055683% | +0.477892% |
| ridge_incumbent | producer2_controller1_easy | +3.896838% | +2.897443% to +4.997470% | +5.391825% |
| old_stop | producer0_controller1_all | +0.000000% | +0.000000% to +0.000000% | +0.000000% |
| old_stop | producer0_controller1_easy | +0.000000% | +0.000000% to +0.000000% | +0.000000% |
| old_stop | producer0_controller2_all | +0.000000% | +0.000000% to +0.000000% | +0.000000% |
| old_stop | producer0_controller2_easy | +0.000000% | +0.000000% to +0.000000% | +0.000000% |
| old_stop | producer1_controller0_all | +0.000000% | +0.000000% to +0.000000% | +0.000000% |
| old_stop | producer1_controller0_easy | +0.000000% | +0.000000% to +0.000000% | +0.000000% |
| old_stop | producer1_controller2_all | +0.000000% | +0.000000% to +0.000000% | +0.000000% |
| old_stop | producer1_controller2_easy | +0.000000% | +0.000000% to +0.000000% | +0.000000% |
| old_stop | producer2_controller0_all | +0.000000% | +0.000000% to +0.000000% | +0.000000% |
| old_stop | producer2_controller0_easy | +0.000000% | +0.000000% to +0.000000% | +0.000000% |
| old_stop | producer2_controller1_all | +0.000000% | +0.000000% to +0.000000% | +0.000000% |
| old_stop | producer2_controller1_easy | +0.000000% | +0.000000% to +0.000000% | +0.000000% |
| previous_matched | producer0_controller1_all | +1.288326% | +0.829198% to +1.767807% | +2.385494% |
| previous_matched | producer0_controller1_easy | +0.146939% | -0.048841% to +0.341022% | -0.024382% |
| previous_matched | producer0_controller2_all | +0.211841% | -0.200776% to +0.777415% | +0.538883% |
| previous_matched | producer0_controller2_easy | -0.038096% | -0.129777% to +0.016224% | -0.010597% |
| previous_matched | producer1_controller0_all | +0.689428% | +0.193477% to +1.408189% | -0.269141% |
| previous_matched | producer1_controller0_easy | +0.040569% | -0.011345% to +0.096147% | -0.026887% |
| previous_matched | producer1_controller2_all | +0.343377% | -0.260496% to +1.346940% | -0.278522% |
| previous_matched | producer1_controller2_easy | +0.081737% | -0.025481% to +0.214376% | +0.016756% |
| previous_matched | producer2_controller0_all | +0.204665% | +0.018444% to +0.436711% | +0.243448% |
| previous_matched | producer2_controller0_easy | -0.000254% | -0.070892% to +0.070014% | -0.002714% |
| previous_matched | producer2_controller1_all | +0.420621% | +0.124431% to +0.757536% | +0.559328% |
| previous_matched | producer2_controller1_easy | +0.128582% | +0.055662% to +0.212245% | +0.141037% |
| raw_neural | producer0_controller1_all | -11.242704% | -16.313436% to -6.882437% | +0.324066% |
| raw_neural | producer0_controller1_easy | -4.706752% | -8.565281% to -1.261784% | +7.041178% |
| raw_neural | producer0_controller2_all | -11.242704% | -16.313436% to -6.882437% | +0.324066% |
| raw_neural | producer0_controller2_easy | -4.706752% | -8.565281% to -1.261784% | +7.041178% |
| raw_neural | producer1_controller0_all | -15.356453% | -20.271710% to -9.771857% | -0.889368% |
| raw_neural | producer1_controller0_easy | -9.315072% | -13.442427% to -4.571110% | +6.320664% |
| raw_neural | producer1_controller2_all | -15.356453% | -20.271710% to -9.771857% | -0.889368% |
| raw_neural | producer1_controller2_easy | -9.315072% | -13.442427% to -4.571110% | +6.320664% |
| raw_neural | producer2_controller0_all | +8.441971% | +4.673792% to +12.055045% | +14.057166% |
| raw_neural | producer2_controller0_easy | +12.859978% | +8.220844% to +17.186019% | +18.418539% |
| raw_neural | producer2_controller1_all | +8.441971% | +4.673792% to +12.055045% | +14.057166% |
| raw_neural | producer2_controller1_easy | +12.859978% | +8.220844% to +17.186019% | +18.418539% |
